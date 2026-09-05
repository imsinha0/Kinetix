"""PPO on the Banyan Kinetix task with a 2-round continual protocol.

Adapted from experiments/ppo.py; the PPO internals (rollout, GAE, update,
RMS obs normalisation, W&B callback, video rendering) are kept as close to
that file as possible. What changes is the seam around the environment:

  * The env is the Banyan combine-zone task (kinetix_banyan.task_layer); env
    params / static params come from ``prepare_banyan_level``, NOT from the
    env_size config groups.
  * Two training rounds: round 1 samples episodes from the d1 task bank,
    round 2 from the d2 bank. Because the reset_fn is baked into the jitted
    env, we build two envs (identical except for the reset_fn) and drive the
    same per-update step function once per round. Network params, optimizer
    state, the RMS obs-normaliser state and the global update counter carry
    across the boundary.
  * Boundary protocol: after round 1's last gradient step we evaluate BOTH
    banks (S_end(d1), S_start(d2)) BEFORE any round-2 gradient step, and
    both again at the very end (S_end(d2), S_end_final(d1)); transfer
    metrics are computed from these four fresh measurements only.
  * Eval replaces the hand-designed Kinetix levels entirely: N fixed
    episodes per (bank x depth), generated with fixed PRNG keys, rolled out
    with a near-deterministic (temperature ~ 0) policy.
"""

import functools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec
from omegaconf import OmegaConf

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
BANYAN_GRID = REPO_ROOT.parent / "banyan-grid"
for _path in (str(REPO_ROOT), str(BANYAN_GRID)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark.baselines.utils.continuous_banyan import build_d1_d2_task_banks

from kinetix.data import get_valid_action_mask
from kinetix.environment.wrappers import LogWrapper
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import (
    RunningMeanStandard,
    general_eval,
    init_wandb,
    load_train_state_from_wandb_artifact_path,
    make_video_fn,
    normalise_config,
    parallel_rms_update,
    rms_init,
    rms_normalise,
    save_model,
)
from kinetix.util.eval_utils import EpisodeMetrics, EvalMetrics, RenderMetrics
from kinetix.util.train_utils import compute_gns_metrics, get_logger, weight_norm

from kinetix_banyan.task_layer import (
    NUM_OBJECT_SLOTS,
    make_banyan_env,
    make_banyan_reset_fn,
    prepare_banyan_level,
)

logger = get_logger()

os.environ["WANDB_DISABLE_SERVICE"] = "True"

RECIPE_LENGTH = 2  # banks contain depth-1 and depth-2 rows; config.task_depths selects the ones used
EVAL_LEVEL_SEEDS = {("d1", 1): 1001, ("d1", 2): 1002, ("d2", 1): 1003, ("d2", 2): 1004}
EVAL_METRIC_NAMES = ("success", "deadend", "timeout", "ep_len", "n_in_zone_end")


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: jnp.ndarray
    valid_action_mask: jnp.ndarray


class RunnerState(NamedTuple):
    train_state: Any
    env_state: Any
    last_obs: Any
    last_done: jnp.ndarray
    extra: dict
    hstate: Any
    rng: jnp.ndarray
    update_step: jnp.ndarray


def get_train_state_from_config(config, rng: jax.Array, env, env_params):
    dummy_batch_dim = 1
    network = make_network_from_config(env, env_params, config)
    rng, _rng = jax.random.split(rng)
    obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, dummy_batch_dim), env_params)
    dones = jnp.zeros((dummy_batch_dim), dtype=jnp.bool_)
    rng, _rng = jax.random.split(rng)
    init_hstate = GeneralActorCriticRNN.initialize_carry(dummy_batch_dim)
    init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
    network_params = {"params": network.init(_rng, init_hstate, init_x)["params"]}

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def linear_warmup_cosine_decay_schedule(count):
        frac = (count // (config["num_minibatches"] * config["update_epochs"])) / config[
            "num_updates"
        ]  # between 0 and 1
        delta = config["peak_lr"] - config["initial_lr"]
        frac_diff_max = 1.0 - config["warmup_frac"]
        frac_cosine = (frac - config["warmup_frac"]) / frac_diff_max

        return jax.lax.select(
            frac < config["warmup_frac"],
            config["initial_lr"] + delta * frac / config["warmup_frac"],
            config["peak_lr"] * jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * ((frac_cosine) % 1.0)))),
        )

    if config["anneal_lr"]:
        lr_to_use = linear_schedule
    elif config["warmup_lr"]:
        lr_to_use = linear_warmup_cosine_decay_schedule
    else:
        lr_to_use = config["lr"]
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(lr_to_use, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    return train_state


def build_bank_level_states(base_state, constants, task_bank, depth, num_episodes, seed, num_distractors=0):
    """Fixed eval reset states: the depth-``depth`` rows of ``task_bank``,
    sampled via the vmapped reset fn with fixed PRNG keys."""
    depths = np.asarray(task_bank["depth"])
    (idx,) = np.nonzero(depths == depth)
    assert idx.size > 0, f"task bank has no depth-{depth} rows"
    sub_bank = {k: jnp.asarray(np.asarray(v)[idx]) for k, v in task_bank.items()}
    reset_fn = make_banyan_reset_fn(base_state, constants, sub_bank, num_distractors=num_distractors)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    return jax.vmap(reset_fn)(keys)


def _bank_eval_keys(level_sets):
    return [
        f"eval/{bank}_{metric}_depth{depth}" for (bank, depth) in level_sets for metric in EVAL_METRIC_NAMES
    ]


def make_bank_eval_fn(eval_env, env_params, config, level_sets):
    """Eval = fixed episodes per (bank x depth), near-deterministic policy.

    The reward is sparse in {-1, 0, +1} and any nonzero reward terminates, so
    over the first episode: success = return > 0, deadend = return < 0,
    timeout = neither (episode hit max_timesteps).
    """

    def eval_fn(rng, train_state, rms):
        obs_fn = (lambda obs: rms_normalise(rms, obs, flatten="auto")) if config["rms_norm"] else None
        metrics = {}
        for (bank, depth), levels in level_sets.items():
            rng, _rng = jax.random.split(rng)
            num_levels = levels.task_id.shape[0]
            _, cum_rewards, _, episode_lengths, infos = general_eval(
                _rng,
                eval_env,
                env_params,
                train_state,
                levels,
                env_params.max_timesteps,
                num_levels,
                keep_states=False,
                observation_preprocessing_fn=obs_fn,
                temperature=config["eval_temperature"],
            )
            success = cum_rewards > 0.5
            deadend = cum_rewards < -0.5
            end_idx = jnp.clip(episode_lengths - 1, 0, env_params.max_timesteps - 1)
            n_in_zone_end = infos["n_in_zone"][end_idx, jnp.arange(num_levels)]
            metrics[f"eval/{bank}_success_depth{depth}"] = success.mean()
            metrics[f"eval/{bank}_deadend_depth{depth}"] = deadend.mean()
            metrics[f"eval/{bank}_timeout_depth{depth}"] = (~success & ~deadend).mean()
            metrics[f"eval/{bank}_ep_len_depth{depth}"] = episode_lengths.mean()
            metrics[f"eval/{bank}_n_in_zone_end_depth{depth}"] = n_in_zone_end.astype(jnp.float32).mean()
        return metrics

    return eval_fn


def make_video_eval_fn(eval_env, env_params, config, video_sets, video_fn):
    """Roll out the fixed video episodes with states kept, and render them
    through the stock make_video_fn pipeline."""

    def video_eval(rng, train_state, rms):
        obs_fn = (lambda obs: rms_normalise(rms, obs, flatten="auto")) if config["rms_norm"] else None
        eval_metrics = {}
        for name, levels in video_sets.items():
            rng, _rng = jax.random.split(rng)
            num_levels = levels.task_id.shape[0]
            states, cum_rewards, done_idx, episode_lengths, _infos = general_eval(
                _rng,
                eval_env,
                env_params,
                train_state,
                levels,
                env_params.max_timesteps,
                num_levels,
                keep_states=True,
                observation_preprocessing_fn=obs_fn,
                temperature=config["eval_temperature"],
            )
            episode_metrics = EpisodeMetrics(
                episode_lengths=episode_lengths,
                episode_returns=cum_rewards,
                episode_solve_rates=(cum_rewards > 0.5).astype(jnp.float32),
            )
            eval_metrics[name] = EvalMetrics(
                episode_metrics=episode_metrics,
                render_metrics=RenderMetrics(
                    states_to_plot=states.env_state,  # unwrap the LogWrapper state
                    episode_metrics=episode_metrics,
                    original_level_list_index=jnp.arange(num_levels),
                ),
            )
        return video_fn(jnp.asarray(True), eval_metrics)

    return video_eval


def _bank_success(metrics, bank, depths):
    """Bank success = mean over the configured task depths."""
    return float(np.mean([float(metrics[f"eval/{bank}_success_depth{d}"]) for d in depths]))


def _filter_bank_depths(bank, depths):
    """Keep only the rows of ``bank`` whose depth is in ``depths``."""
    keep = np.isin(np.asarray(bank["depth"]), np.asarray(depths))
    assert keep.any(), f"task bank has no rows at depths {depths}"
    return {k: jnp.asarray(np.asarray(v)[keep]) for k, v in bank.items()}


def make_train(config, env_params, static_env_params, envs, full_sets, periodic_sets, video_sets):
    NUM_GPUS = jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=["devices"])

    replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    partitioned_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("devices"))

    config["num_gpus"] = NUM_GPUS
    assert config["num_train_envs"] % NUM_GPUS == 0
    config["num_train_envs"] = config["num_train_envs"] // NUM_GPUS

    # The reset_fn is bypassed during eval (the fixed levels are passed as
    # override reset states), so a single eval env suffices.
    eval_env = envs[1]
    periodic_eval_fn = make_bank_eval_fn(eval_env, env_params, config, periodic_sets)
    periodic_eval_keys = _bank_eval_keys(periodic_sets)
    full_eval_fn = jax.jit(make_bank_eval_fn(eval_env, env_params, config, full_sets))

    render_static_env_params = static_env_params.replace(downscale=1, screen_dim=(125, 125))
    render_env_params = env_params.replace(pixels_per_unit=25)
    pixel_renderer = jax.jit(make_render_pixels(render_env_params, render_static_env_params))
    pixel_render_fn = lambda x: pixel_renderer(x) / 255.0
    video_fn = make_video_fn(render_env_params, render_static_env_params, pixel_render_fn)
    video_eval_fn = jax.jit(make_video_eval_fn(eval_env, env_params, config, video_sets, video_fn))

    time_start = time.time()
    last_time = time.time()

    def _maybe_normalise(
        rms: RunningMeanStandard, obs: jnp.ndarray, should_update=True
    ) -> tuple[RunningMeanStandard, jnp.ndarray]:
        if config["rms_norm"]:
            new_obs = rms_normalise(rms, obs, flatten="auto")
            if should_update:
                rms = parallel_rms_update(rms, obs)
        else:
            new_obs = obs

        return rms, new_obs

    def _global_env_steps(update_step):
        return int(update_step) * int(config["num_steps"]) * int(config["num_train_envs"]) * int(NUM_GPUS)

    runner_state_spec = RunnerState(
        train_state=PartitionSpec(),  # replicated
        env_state=PartitionSpec("devices"),
        last_obs=PartitionSpec("devices"),
        last_done=PartitionSpec("devices"),
        extra=PartitionSpec(),  # rms norm: replicated
        hstate=PartitionSpec("devices"),
        rng=PartitionSpec("devices"),
        update_step=PartitionSpec(),  # replicated
    )

    def make_init_fn(env):
        def _init(rng):
            rng = rng.squeeze(0)
            obsv, env_state = jax.vmap(env.reset, (0, None))(
                jax.random.split(rng, config["num_train_envs"]), env_params
            )
            init_hstate = GeneralActorCriticRNN.initialize_carry(config["num_train_envs"])
            init_dones = jnp.zeros((config["num_train_envs"]), dtype=bool)

            return obsv, env_state, init_hstate, init_dones

        return jax.jit(
            jax.shard_map(
                _init,
                mesh=mesh,
                in_specs=(PartitionSpec("devices"),),
                out_specs=PartitionSpec("devices"),
                check_vma=False,
            )
        )

    def make_update_step(env, round_idx):
        # Copied from ppo.py's _update_step; deltas: the env is round-specific,
        # eval is the banyan (bank x depth) eval, and the callback logs banyan
        # train diagnostics plus a `round` label.
        def _update_step(runner_state, unused):
            # squeeze the RNG (undoes expand_dims added at the end of the previous step for sharding)
            runner_state = runner_state._replace(rng=runner_state.rng.squeeze(0))

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    extra,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state

                extra["rms"], obs_to_use = _maybe_normalise(extra["rms"], last_obs, should_update=True)
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = train_state.apply_fn(train_state.params, hstate, ac_in)
                obs_to_save = last_obs

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)

                valid_action_mask = get_valid_action_mask(env_state, static_env_params, action)

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["num_train_envs"]),
                    env_state,
                    action,
                    env_params,
                )

                transition = Transition(
                    last_done, action, value, reward, log_prob, obs_to_save, info, valid_action_mask=valid_action_mask
                )
                runner_state = RunnerState(train_state, env_state, obsv, done, extra, hstate, rng, update_step)
                return runner_state, transition

            initial_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                extra,
                hstate,
                rng,
                update_step,
            ) = runner_state
            _, obs_to_use = _maybe_normalise(extra["rms"], last_obs, should_update=False)
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            )
            _, _, last_val = train_state.apply_fn(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - next_done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, obs_to_use = jax.vmap(
                            functools.partial(_maybe_normalise, should_update=False),
                            (None, 0),
                        )(extra["rms"], traj_batch.obs)
                        _, pi, value = train_state.apply_fn(params, init_hstate[0], (obs_to_use, traj_batch.done))

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        entropy_disentangled = pi.entropy_disentangled()
                        action_mask = traj_batch.valid_action_mask
                        entropy_disentangled_mean = (
                            (entropy_disentangled * action_mask).sum(axis=-1) / jnp.maximum(1, action_mask.sum(axis=-1))
                        ).mean()

                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy

                        return total_loss, {
                            "loss/value": value_loss,
                            "loss/entropy": entropy,
                            "loss/actor": loss_actor,
                            "loss/total": total_loss,
                            "metrics/entropy_disentangled": entropy_disentangled_mean,
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    total_loss = jax.lax.pmean(total_loss, axis_name="devices")
                    g_squared, s, grad_norm, grads = compute_gns_metrics(grads, advantages.shape[0])
                    total_loss = total_loss[0], total_loss[1] | {
                        "metrics/g_squared": jnp.mean(g_squared),
                        "metrics/s": jnp.mean(s),
                        "metrics/grad_norm": jnp.mean(grad_norm),
                    }

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["num_train_envs"])
                batch = (init_hstate, traj_batch, advantages, targets)

                if config["full_minibatch_shuffle"]:
                    assert not config["recurrent_model"], "Full minibatch shuffle only works with non-recurrent models"

                    # Properly shuffle across ALL dimensions (steps and envs)
                    num_elements = config["num_steps"] * config["num_train_envs"]
                    full_permutation = jax.random.permutation(_rng, num_elements)

                    def reshuffle(x):
                        # x has shape (num_steps, num_train_envs, ...)
                        orig_shape = x.shape
                        x = x.reshape((num_elements,) + orig_shape[2:])
                        x = jnp.take(x, full_permutation, axis=0)
                        x = x.reshape(orig_shape)
                        return x

                    # We don't shuffle init_hstate because it's per-env, but it's irrelevant for non-recurrent models anyway.
                    # We shuffle traj_batch, advantages, and targets.
                    shuffled_traj_batch = jax.tree.map(reshuffle, traj_batch)
                    shuffled_advantages = reshuffle(advantages)
                    shuffled_targets = reshuffle(targets)

                    shuffled_batch = (init_hstate, shuffled_traj_batch, shuffled_advantages, shuffled_targets)
                else:
                    shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = jax.tree.map(lambda x: x[None, :], initial_hstate)
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            metrics_to_log = jax.tree.map(lambda x: x.mean(), loss_info[1])
            metrics_to_log["metrics/gns"] = metrics_to_log["metrics/s"] / metrics_to_log["metrics/g_squared"]
            rng = update_state[-1]

            if config["use_wandb"]:
                param_metrics = {"metrics/param_norm": weight_norm(train_state.params)}

                def _real_eval(rng):
                    eval_metrics = periodic_eval_fn(rng, runner_state.train_state, runner_state.extra["rms"])
                    return eval_metrics, jnp.asarray(True)

                def _fake_eval(rng):
                    return {k: jnp.zeros(()) for k in periodic_eval_keys}, jnp.asarray(False)

                rng, _rng = jax.random.split(rng)
                should_eval = jnp.logical_and(config["eval_freq"] > 0, update_step % config["eval_freq"] == 0)
                eval_metrics, should_log_evals = jax.lax.cond(should_eval, _real_eval, _fake_eval, _rng)

                def callback(raw_info, update_step, eval_metrics, should_log_evals, metrics_to_log):
                    nonlocal last_time
                    time_now = time.time()
                    delta_time = time_now - last_time
                    last_time = time_now
                    dones = raw_info["returned_episode"]
                    denom = jnp.maximum(1, dones.sum())
                    to_log = {
                        "episode_return": (raw_info["returned_episode_returns"] * dones).sum() / denom,
                        "episode_solved": (raw_info["returned_episode_solved"] * dones).sum() / denom,
                        "episode_length": (raw_info["returned_episode_lengths"] * dones).sum() / denom,
                        "num_completed_episodes": dones.sum(),
                        # Banyan train diagnostics (all at episode end)
                        "train/success_rate": (raw_info["GoalR"] * dones).sum() / denom,
                        "train/deadend_rate": (raw_info["deadend"] * dones).sum() / denom,
                        "train/episode_length": (raw_info["returned_episode_lengths"] * dones).sum() / denom,
                        "train/n_in_zone": (raw_info["n_in_zone"] * dones).sum() / denom,
                        "train/n_required_in_zone": (raw_info["n_required_in_zone"] * dones).sum() / denom,
                        "train/task_depth": (raw_info["task_depth"] * dones).sum() / denom,
                        **metrics_to_log,
                    }
                    to_log["round"] = round_idx
                    to_log["timing/num_updates"] = update_step
                    to_log["timing/num_model_forward_passes"] = int(update_step) * (
                        int(config["num_steps"]) + config["num_minibatches"] * config["update_epochs"]
                    )
                    to_log["timing/num_model_backward_passes"] = int(update_step) * (
                        config["num_minibatches"] * config["update_epochs"]
                    )
                    to_log["timing/num_env_steps"] = (
                        int(update_step) * int(config["num_steps"]) * int(config["num_train_envs"]) * int(NUM_GPUS)
                    )
                    to_log["timing/sps"] = (
                        int(config["num_steps"]) * int(config["num_train_envs"]) * int(NUM_GPUS)
                    ) / delta_time
                    to_log["timing/sps_agg"] = (to_log["timing/num_env_steps"]) / (time_now - time_start)

                    if metrics_to_log["device_id"] != 0:
                        return
                    if should_log_evals:
                        to_log |= {k: float(v) for k, v in eval_metrics.items()}
                    wandb.log(to_log)

                metrics_to_log = jax.lax.pmean(metrics_to_log, axis_name="devices")
                metrics_to_log["device_id"] = jax.lax.axis_index("devices")
                metrics_to_log |= param_metrics
                jax.debug.callback(
                    callback, traj_batch.info, update_step, eval_metrics, should_log_evals, metrics_to_log
                )

            runner_state = RunnerState(
                train_state,
                env_state,
                last_obs,
                last_done,
                extra,
                hstate,
                jnp.expand_dims(rng, axis=0),
                update_step + 1,
            )
            return runner_state, metric

        return jax.jit(
            jax.shard_map(
                _update_step,
                mesh=mesh,
                in_specs=(
                    runner_state_spec,
                    PartitionSpec(),
                ),
                out_specs=(runner_state_spec, PartitionSpec()),
                check_vma=False,
            )
        )

    def _make_run_round(update_fn, num_updates):
        def run(runner_state):
            return jax.lax.scan(update_fn, runner_state, None, num_updates)

        return jax.jit(run)

    init_fns = {r: make_init_fn(envs[r]) for r in (1, 2)}
    run_round_fns = {r: _make_run_round(make_update_step(envs[r], r), config[f"num_updates_d{r}"]) for r in (1, 2)}

    def _log_full_eval(metrics, extra_metrics, update_step, round_idx):
        to_log = {k: float(v) for k, v in metrics.items()}
        to_log |= extra_metrics
        to_log["round"] = round_idx
        to_log["timing/num_env_steps"] = _global_env_steps(update_step)
        if jax.process_index() == 0:
            logger.info(f"Full eval (round {round_idx}): { {k: round(v, 4) for k, v in to_log.items()} }")
            if config["use_wandb"]:
                wandb.log(to_log)

    def _log_videos(videos, round_idx, update_step):
        # Same rendering path as create_eval_metrics_dict_for_logging, but
        # keyed as video/round{r}/... (see task protocol).
        if not config["use_wandb"] or jax.process_index() != 0:
            return
        to_log = {"round": round_idx, "timing/num_env_steps": _global_env_steps(update_step)}
        for name, video in videos.items():
            frames_all = np.asarray(video.frames)
            for i in range(frames_all.shape[1]):
                length = max(int(video.episode_metrics.episode_lengths[i]), 1)
                frames = frames_all[:length, i]
                caption = (
                    f"R = {float(video.episode_metrics.episode_returns[i]):.2f} | "
                    f"L = {int(video.episode_metrics.episode_lengths[i])} | "
                    f"S = {float(video.episode_metrics.episode_solve_rates[i]) > 0}"
                )
                np_vid = frames.transpose(0, 3, 2, 1)[:, :, ::-1, :]
                to_log[f"video/round{round_idx}/{name}_{i}"] = wandb.Video(
                    (np_vid * 255).astype(np.uint8), fps=15, format="gif", caption=caption
                )
        wandb.log(to_log)

    def _adapt_tree(src_tree, tgt_tree, what):
        """Copy pretrained leaves onto the target tree, zero-padding any leaf
        that differs from the target in exactly ONE axis (the widened circle
        feature dim: 19 -> 19+TYPE_CODE_DIM). Zero rows for new input features
        leave the pretrained function exactly unchanged; 'var'-named leaves pad
        with 1.0. Anything else falls back to the fresh init with a warning."""
        import flax.traverse_util as trav

        flat_src = trav.flatten_dict(src_tree)
        flat_tgt = trav.flatten_dict(tgt_tree)
        out = {}
        n_exact = n_padded = n_fresh = 0
        for key, tv in flat_tgt.items():
            sv = flat_src.get(key)
            if sv is not None and sv.shape == tv.shape:
                out[key] = sv
                n_exact += 1
                continue
            padded = None
            if sv is not None and sv.ndim == tv.ndim:
                diff_axes = [a for a in range(sv.ndim) if sv.shape[a] != tv.shape[a]]
                if len(diff_axes) == 1 and tv.shape[diff_axes[0]] > sv.shape[diff_axes[0]]:
                    ax = diff_axes[0]
                    pad_shape = list(sv.shape)
                    pad_shape[ax] = tv.shape[ax] - sv.shape[ax]
                    fill = 1.0 if "var" in "/".join(map(str, key)).lower() else 0.0
                    padded = jnp.concatenate(
                        [jnp.asarray(sv), jnp.full(pad_shape, fill, dtype=sv.dtype)], axis=ax
                    )
            if padded is not None:
                out[key] = padded
                n_padded += 1
            else:
                out[key] = tv
                n_fresh += 1
                logger.info(f"[pretrained] fresh init for {what}:{'/'.join(map(str, key))}")
        logger.info(f"[pretrained] {what}: {n_exact} exact, {n_padded} padded, {n_fresh} fresh")
        return trav.unflatten_dict(out)

    def train(rng):
        # INIT NETWORK (both rounds share obs/action spaces; use the d1 env)
        rng, _rng = jax.random.split(rng)
        train_state = get_train_state_from_config(config, _rng, envs[1], env_params)

        pretrained_extra_raw = None
        if config.get("pretrained_checkpoint"):
            from kinetix.util.saving import load_params as _load_params

            path = config["pretrained_checkpoint"]
            logger.info(f"[pretrained] loading {path} with shape-padding adaptation")
            all_dict = _load_params(path)
            train_state = train_state.replace(
                params=_adapt_tree(all_dict["params"], train_state.params, "params")
            )
            pretrained_extra_raw = all_dict.get("extra")

        new_extra = None
        if config["load_from_checkpoint"] is not None:
            logger.info(
                "Loading checkpoint from %s (load_only_params=%s)",
                config["load_from_checkpoint"],
                config["load_only_params"],
            )
            train_state, new_extra = load_train_state_from_wandb_artifact_path(
                train_state,
                config["load_from_checkpoint"],
                load_only_params=config["load_only_params"],
                return_extra=True,
                specific_dir="/tmp",
            )
            assert new_extra is not None

        # INIT ROUND-1 ENVS
        rng, _rng = jax.random.split(rng)
        rngs_per_device = jax.device_put(jax.random.split(_rng, NUM_GPUS), partitioned_sharding)
        obsv, env_state, init_hstate, init_dones = init_fns[1](rngs_per_device)

        initial_extra = new_extra if new_extra is not None else {"rms": rms_init(jax.tree.map(lambda x: x[0], obsv))}
        if pretrained_extra_raw is not None and new_extra is None:
            initial_extra = _adapt_tree(pretrained_extra_raw, initial_extra, "extra")

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=jax.device_put(train_state, replicated_sharding),
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            extra=jax.device_put(initial_extra, replicated_sharding),
            hstate=init_hstate,
            rng=jax.random.split(_rng, NUM_GPUS),
            update_step=jax.device_put(jnp.array(0), replicated_sharding),
        )

        # Boundary protocol: the recorded event order is asserted at the end,
        # and update-step counters are asserted at every stage.
        protocol = []

        # ROUND 1: train on the d1 bank
        logger.info(f"Round 1: {config['num_updates_d1']} updates on the d1 bank")
        runner_state, _ = run_round_fns[1](runner_state)
        jax.block_until_ready(runner_state.train_state.params)
        assert int(runner_state.update_step) == config["num_updates_d1"]
        protocol.append("round1_train")

        # BOUNDARY: evaluate BOTH banks with the end-of-round-1 policy. This
        # runs (and is synchronised) before the round-2 runner state exists,
        # so no round-2 gradient step can precede it.
        rng, _rng, _rng_v = jax.random.split(rng, 3)
        boundary_metrics = jax.device_get(full_eval_fn(_rng, runner_state.train_state, runner_state.extra["rms"]))
        protocol.append("boundary_eval")
        assert int(runner_state.update_step) == config["num_updates_d1"], "round-2 update ran before the boundary eval"
        S_end_d1 = _bank_success(boundary_metrics, "d1", config["task_depths"])
        S_start_d2 = _bank_success(boundary_metrics, "d2", config["task_depths"])
        _log_full_eval(
            boundary_metrics,
            {"boundary/S_end_d1": S_end_d1, "boundary/S_start_d2": S_start_d2},
            runner_state.update_step,
            round_idx=1,
        )
        boundary_videos = jax.device_get(video_eval_fn(_rng_v, runner_state.train_state, runner_state.extra["rms"]))
        _log_videos(boundary_videos, 1, runner_state.update_step)

        # ROUND 2: fresh envs from the d2 bank; params, optimizer state, RMS
        # state and the global update counter carry over unchanged.
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        rngs_per_device = jax.device_put(jax.random.split(_rng, NUM_GPUS), partitioned_sharding)
        obsv2, env_state2, init_hstate2, init_dones2 = init_fns[2](rngs_per_device)
        runner_state = RunnerState(
            train_state=runner_state.train_state,  # carried over
            env_state=env_state2,
            last_obs=obsv2,
            last_done=init_dones2,
            extra=runner_state.extra,  # carried over (RMS obs normaliser)
            hstate=init_hstate2,
            rng=jax.random.split(_rng2, NUM_GPUS),
            update_step=runner_state.update_step,  # global step keeps counting
        )
        protocol.append("round2_init")

        logger.info(f"Round 2: {config['num_updates_d2']} updates on the d2 bank")
        runner_state, _ = run_round_fns[2](runner_state)
        jax.block_until_ready(runner_state.train_state.params)
        assert int(runner_state.update_step) == config["num_updates_d1"] + config["num_updates_d2"]
        protocol.append("round2_train")

        # FINAL: evaluate BOTH banks again
        rng, _rng, _rng_v = jax.random.split(rng, 3)
        final_metrics = jax.device_get(full_eval_fn(_rng, runner_state.train_state, runner_state.extra["rms"]))
        protocol.append("final_eval")

        S_end_d2 = _bank_success(final_metrics, "d2", config["task_depths"])
        S_end_final_d1 = _bank_success(final_metrics, "d1", config["task_depths"])
        transfer = {
            "final/S_end_d2": S_end_d2,
            "final/S_end_final_d1": S_end_final_d1,
            "transfer/delta_2": S_end_d1 - S_start_d2,
            "transfer/B_2_1": S_end_final_d1 - S_end_d1,
        }
        for depth in config["task_depths"]:
            transfer[f"transfer/delta_2_depth{depth}"] = float(
                boundary_metrics[f"eval/d1_success_depth{depth}"]
            ) - float(boundary_metrics[f"eval/d2_success_depth{depth}"])
            transfer[f"transfer/B_2_1_depth{depth}"] = float(final_metrics[f"eval/d1_success_depth{depth}"]) - float(
                boundary_metrics[f"eval/d1_success_depth{depth}"]
            )
        _log_full_eval(final_metrics, transfer, runner_state.update_step, round_idx=2)
        final_videos = jax.device_get(video_eval_fn(_rng_v, runner_state.train_state, runner_state.extra["rms"]))
        _log_videos(final_videos, 2, runner_state.update_step)

        assert protocol == ["round1_train", "boundary_eval", "round2_init", "round2_train", "final_eval"], protocol

        if config["save_policy"] and jax.process_index() == 0:
            save_model(
                runner_state.train_state,
                _global_env_steps(runner_state.update_step),
                config,
                is_final=True,
                save_to_wandb=config["use_wandb"],
                extra=jax.device_get(runner_state.extra),
            )

        return {
            "runner_state": runner_state,
            "boundary_metrics": boundary_metrics,
            "final_metrics": final_metrics,
            "transfer": transfer,
        }

    return train


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_banyan")
def main(config):
    name = "PPO-Banyan"
    config = OmegaConf.to_container(config)
    config["learning"]["total_timesteps"] = int(config["total_timesteps_d1"]) + int(config["total_timesteps_d2"])
    config = normalise_config(config, name)

    # Build the task banks once; both rounds and all evals share them.
    d1_bank, d2_bank, meta = build_d1_d2_task_banks(
        d1_num_instances=config["d1_num_instances"],
        d2_num_instances=config["d2_num_instances"],
        num_leaf_slots=NUM_OBJECT_SLOTS,
        id_vocab_size=config["id_vocab_size"],
        recipe_length=RECIPE_LENGTH,
        seed=config["task_seed"],
        task_bank_mode="banyan_items_curriculum",
        object_condition=config["object_condition"],
    )
    overlap = int(meta["round_instance_overlap_count"])
    assert overlap == 0, f"d1/d2 instance overlap must be 0, got {overlap}"

    # Restrict both banks to the configured task depths (e.g. [1] = single
    # object, move-it-to-the-zone tasks only). Training resets, evals, the
    # bank success aggregate and the transfer metrics all use these depths.
    depths_used = tuple(int(d) for d in config["task_depths"])
    assert depths_used and all(1 <= d <= RECIPE_LENGTH for d in depths_used), depths_used
    config["task_depths"] = list(depths_used)
    d1_bank = _filter_bank_depths(d1_bank, depths_used)
    d2_bank = _filter_bank_depths(d2_bank, depths_used)
    bank_depth_sets = tuple((bank, d) for bank in ("d1", "d2") for d in depths_used)

    d1_goals = np.unique(np.asarray(d1_bank["goal_token"]))
    d2_goals = np.unique(np.asarray(d2_bank["goal_token"]))
    config["diversity/n_d1"] = int(np.asarray(d1_bank["depth"]).shape[0])
    config["diversity/n_d2"] = int(np.asarray(d2_bank["depth"]).shape[0])
    config["diversity/d1_d2_overlap"] = overlap
    config["diversity/d1_distinct_goal_tokens"] = int(d1_goals.size)
    config["diversity/d2_distinct_goal_tokens"] = int(d2_goals.size)
    config["diversity/d1_d2_goal_token_overlap"] = int(np.intersect1d(d1_goals, d2_goals).size)
    config["diversity/round_leaf_overlap_count"] = int(meta["round_leaf_overlap_count"])
    logger.info(
        f"[BANKS] depths={depths_used} n_d1={config['diversity/n_d1']} n_d2={config['diversity/n_d2']} "
        f"distinct goal tokens d1={d1_goals.size} d2={d2_goals.size} "
        f"goal-token overlap={config['diversity/d1_d2_goal_token_overlap']}"
    )

    rulebook = meta["global_rulebook"]
    base_state, static_env_params, env_params, constants = prepare_banyan_level(
        merge_lhs=np.asarray(rulebook["merge_lhs"]),
        merge_rhs=np.asarray(rulebook["merge_rhs"]),
        token_vocab_size=int(meta["token_vocab_size"]),
        lip_height=float(config.get("lip_height", 0.55)),
    )
    if config.get("max_timesteps") is not None:
        # Horizon deviation from the stock level (256): the lipped two-delivery
        # chain does not fit 256 steps. Logged loudly via config.
        env_params = env_params.replace(max_timesteps=int(config["max_timesteps"]))
        logger.info(f"[HORIZON OVERRIDE] max_timesteps -> {env_params.max_timesteps}")
    assert float(env_params.dense_reward_scale) == 0.0, "Banyan reward must be sparse (dense_reward_scale == 0)"
    # Null-policy floor for this level version (measured offline via
    # kinetix_banyan.lip_tune): stamped so figures can draw the floor line.
    config["null_policy_floor"] = float(config.get("null_policy_floor", -1.0))
    config["lip_height_used"] = float(config.get("lip_height", 0.55))
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    config["kinetix_version"] = "v3.0.0"
    try:
        config["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        config["git_sha"] = "unknown"

    steps_per_update = int(config["num_steps"]) * int(config["num_train_envs"])
    config["num_updates_d1"] = int(config["total_timesteps_d1"]) // steps_per_update
    config["num_updates_d2"] = int(config["total_timesteps_d2"]) // steps_per_update
    assert config["num_updates_d1"] > 0 and config["num_updates_d2"] > 0

    def _make_env(task_bank, level_base=None, level_constants=None):
        lb = base_state if level_base is None else level_base
        lc = constants if level_constants is None else level_constants
        reset_fn = make_banyan_reset_fn(lb, lc, task_bank, num_distractors=int(config["num_distractors"]))
        return LogWrapper(
            make_banyan_env(
                base_state=lb,
                static_env_params=static_env_params,
                env_params=env_params,
                constants=lc,
                reset_fn=reset_fn,
                action_type=config["action_type_str"],
                reward_first_lift=float(config.get("reward_first_lift", 0.0)),
                reward_first_deposit=float(config.get("reward_first_deposit", 0.0)),
                reward_height_scale=float(config.get("reward_height_scale", 0.0)),
            )
        )

    # Lip curriculum: round 1 may train on a different lip height (e.g. 0 =
    # no lip) while ALL evals and round 2 use the final lipped level. The
    # final level is the task; round 1's easier level is exploration scaffold.
    lip_r1 = float(config.get("lip_height_round1", config.get("lip_height", 0.55)))
    if lip_r1 != float(config.get("lip_height", 0.55)):
        base_r1, _static_r1, _ep_r1, constants_r1 = prepare_banyan_level(
            merge_lhs=np.asarray(rulebook["merge_lhs"]),
            merge_rhs=np.asarray(rulebook["merge_rhs"]),
            token_vocab_size=int(meta["token_vocab_size"]),
            lip_height=lip_r1,
        )
        logger.info(f"[LIP CURRICULUM] round-1 lip={lip_r1}, round-2/evals lip={config.get('lip_height', 0.55)}")
        envs = {1: _make_env(d1_bank, base_r1, constants_r1), 2: _make_env(d2_bank)}
    else:
        envs = {1: _make_env(d1_bank), 2: _make_env(d2_bank)}

    # Fixed eval episodes per (bank x depth); the periodic/video sets are
    # fixed-prefix subsets of the full (boundary/final) sets.
    banks = {"d1": d1_bank, "d2": d2_bank}
    assert config["eval_num_episodes_periodic"] <= config["eval_num_episodes_full"]
    assert config["eval_num_video_episodes"] <= config["eval_num_episodes_full"]
    full_sets = {
        (bank, depth): build_bank_level_states(
            base_state,
            constants,
            banks[bank],
            depth,
            config["eval_num_episodes_full"],
            EVAL_LEVEL_SEEDS[(bank, depth)],
            num_distractors=int(config["num_distractors"]),
        )
        for (bank, depth) in bank_depth_sets
    }
    periodic_sets = {
        k: jax.tree.map(lambda x: x[: config["eval_num_episodes_periodic"]], v) for k, v in full_sets.items()
    }
    video_sets = {
        f"{bank}_depth{depth}": jax.tree.map(lambda x: x[: config["eval_num_video_episodes"]], full_sets[(bank, depth)])
        for (bank, depth) in bank_depth_sets
    }

    if config["use_wandb"]:
        if jax.process_index() == 0:
            init_wandb(config, name, settings=wandb.Settings(quiet=True))
        else:
            os.environ["WANDB_MODE"] = "disabled"

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    train = make_train(config, env_params, static_env_params, envs, full_sets, periodic_sets, video_sets)
    train(_rng)


if __name__ == "__main__":
    main()
