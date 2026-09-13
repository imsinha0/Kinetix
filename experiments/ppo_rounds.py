"""PPO on procedurally generated Kinetix locomotion tasks with an N-round continual protocol.

Kinetix's `sample_locomotion_level` (kinetix/environment/ued/locomotion_distribution.py)
generates walkers (body + legs + optional ankles + extra legs, random dimensions,
friction, density, motor bindings) that must reach a goal circle placed at a random
x-position. Every level is the same kind of task, so difficulty is similar by
construction and a PRNG key fully specifies a task.

Protocol (Banyan paper, Figure 6 style):
  * `num_rounds` rounds of `steps_per_round` env steps each. Round r trains on its own
    task pool: `tasks_per_round` distinct level keys, key(r, i) = fold_in(fold_in(seed, r), i),
    so pools are disjoint across rounds and reproducible. `tasks_per_round` is the
    diversity axis (1 = one fixed level for the whole round; 65536 ≈ a fresh level every episode).
  * Every pool has a fixed evaluation set (first `eval_num_episodes_*` keys, cycled).
    All pools are evaluated periodically (giving S(d_j)(t) for every j throughout training)
    and in full at every round boundary. From the boundary matrix M[b][j] (success on pool j
    after round b) we log S_end(d_r) = M[r][r], the forward gap Δ_r = M[r-1][r-1] - M[r-1][r],
    and backward transfer B(R, j) = M[R][j] - M[j][j].
  * GIFs: fixed video episodes from pool 1 and from the current round's pool, logged every
    `video_freq_updates` updates and at every boundary.
  * PPO internals (rollout, GAE, update, RMS normalisation) are copied from ppo.py/ppo_banyan.py
    unchanged; network, optimizer state and RMS state carry across rounds.
"""

import functools
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import serialization
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec
from omegaconf import OmegaConf

from kinetix.data import get_valid_action_mask
from kinetix.environment.ued.locomotion_distribution import DEFAULT_UED_PARAMS, sample_locomotion_level
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import (
    RunningMeanStandard,
    general_eval,
    generate_params_from_config,
    init_wandb,
    make_video_fn,
    normalise_config,
    parallel_rms_update,
    rms_init,
    rms_normalise,
    save_model,
)
from kinetix.util.eval_utils import EpisodeMetrics, EvalMetrics, RenderMetrics
from kinetix.util.train_utils import compute_gns_metrics, get_logger, make_env, weight_norm

logger = get_logger()

os.environ["WANDB_DISABLE_SERVICE"] = "True"

EVAL_METRIC_NAMES = ("success", "ep_len")


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
        optax.zero_nans(),  # a NaN gradient (e.g. from a physics blow-up) skips the update instead of killing the run
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(lr_to_use, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    return train_state


# ----------------------------------------------------------------------------------
# Task pools
# ----------------------------------------------------------------------------------
def make_task_key_fn(task_seed: int, round_idx: int):
    """key(round, i): the PRNG key that fully specifies task i of the round's pool."""
    base = jax.random.PRNGKey(int(task_seed))
    round_key = jax.random.fold_in(base, int(round_idx))

    def task_key(i):
        return jax.random.fold_in(round_key, i)

    return task_key


def _locomotion_kwargs(config):
    """kwargs for sample_locomotion_level. The special key ``ued`` (dict) overrides fields of
    the locomotion UEDParams, e.g. {floor_prob_red: 0.0, floor_prob_normal: 1.0} for no lava floors."""
    kw = dict(config.get("locomotion_kwargs") or {})
    ued = kw.pop("ued", None)
    if ued:
        kw["ued_params"] = DEFAULT_UED_PARAMS.replace(**{k: float(v) for k, v in dict(ued).items()})
    return kw


def make_pool_reset_fn(config, env_params, static_env_params, round_idx: int):
    """Reset fn for round ``round_idx``: sample one of the pool's ``tasks_per_round`` levels."""
    n = int(config["tasks_per_round"])
    kw = _locomotion_kwargs(config)
    task_key = make_task_key_fn(config["task_seed"], round_idx)

    def reset(rng):
        i = jax.random.randint(rng, (), 0, n)
        return sample_locomotion_level(task_key(i), env_params, static_env_params, **kw)

    return reset


def build_pool_levels(config, env_params, static_env_params, round_idx: int, num_episodes: int):
    """Fixed eval levels for a pool: tasks 0..min(n, num_episodes)-1, cycled to num_episodes."""
    n = int(config["tasks_per_round"])
    kw = _locomotion_kwargs(config)
    task_key = make_task_key_fn(config["task_seed"], round_idx)
    idx = jnp.arange(num_episodes) % n

    def _level(i):
        return sample_locomotion_level(task_key(i), env_params, static_env_params, **kw)

    return jax.vmap(_level)(idx)


def _num_levels(levels):
    return jax.tree.leaves(levels)[0].shape[0]


def _pool_eval_keys(num_rounds):
    return [f"eval/pool{j}_{m}" for j in range(1, num_rounds + 1) for m in EVAL_METRIC_NAMES]


def make_pool_eval_fn(eval_env, env_params, config, pool_levels):
    """Eval = fixed episodes per pool, near-deterministic policy. Success = the goal was
    reached (info GoalR) before the episode ended."""

    def eval_fn(rng, train_state, rms):
        obs_fn = (lambda obs: rms_normalise(rms, obs, flatten="auto")) if config["rms_norm"] else None
        metrics = {}
        for j, levels in pool_levels.items():
            rng, _rng = jax.random.split(rng)
            num_levels = _num_levels(levels)
            _, _cum_rewards, _, episode_lengths, infos = general_eval(
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
            mask = jnp.arange(env_params.max_timesteps)[:, None] < episode_lengths[None, :]
            success = jnp.any(infos["GoalR"] & mask, axis=0)
            metrics[f"eval/pool{j}_success"] = success.mean()
            metrics[f"eval/pool{j}_ep_len"] = episode_lengths.mean()
        return metrics

    return eval_fn


def make_video_eval_fn(eval_env, env_params, config, video_levels_by_pool, video_fn):
    """Roll out the fixed video episodes of pool 1 and of the current pool (traced index),
    keeping states, and render them through the stock make_video_fn pipeline."""
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[video_levels_by_pool[j] for j in sorted(video_levels_by_pool)])

    def video_eval(rng, train_state, rms, cur_pool_index):
        obs_fn = (lambda obs: rms_normalise(rms, obs, flatten="auto")) if config["rms_norm"] else None
        sets = {
            "pool1": jax.tree.map(lambda x: x[0], stacked),
            "current": jax.tree.map(lambda x: x[cur_pool_index], stacked),
        }
        eval_metrics = {}
        for name, levels in sets.items():
            rng, _rng = jax.random.split(rng)
            num_levels = _num_levels(levels)
            states, _cum_rewards, done_idx, episode_lengths, infos = general_eval(
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
            mask = jnp.arange(env_params.max_timesteps)[:, None] < episode_lengths[None, :]
            success = jnp.any(infos["GoalR"] & mask, axis=0).astype(jnp.float32)
            episode_metrics = EpisodeMetrics(
                episode_lengths=episode_lengths,
                episode_returns=_cum_rewards,
                episode_solve_rates=success,
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


def make_train(config, env_params, static_env_params, envs, full_pools, periodic_pools, video_pools):
    NUM_GPUS = jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=["devices"])

    replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    partitioned_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("devices"))

    config["num_gpus"] = NUM_GPUS
    assert config["num_train_envs"] % NUM_GPUS == 0
    config["num_train_envs"] = config["num_train_envs"] // NUM_GPUS
    num_rounds = int(config["num_rounds"])

    # The reset_fn is bypassed during eval (fixed levels are passed as override reset
    # states), so a single eval env suffices.
    eval_env = envs[1]
    periodic_eval_fn = make_pool_eval_fn(eval_env, env_params, config, periodic_pools)
    periodic_eval_keys = _pool_eval_keys(num_rounds)
    full_eval_fn = jax.jit(make_pool_eval_fn(eval_env, env_params, config, full_pools))

    render_static_env_params = static_env_params.replace(downscale=1, screen_dim=(125, 125))
    render_env_params = env_params.replace(pixels_per_unit=25)
    pixel_renderer = jax.jit(make_render_pixels(render_env_params, render_static_env_params))
    pixel_render_fn = lambda x: pixel_renderer(x) / 255.0
    video_fn = make_video_fn(render_env_params, render_static_env_params, pixel_render_fn)
    video_eval_fn = jax.jit(make_video_eval_fn(eval_env, env_params, config, video_pools, video_fn))

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
        # Copied from ppo.py's _update_step (via ppo_banyan.py); deltas: the env is
        # round-specific, eval covers all task pools, and the callback logs a `round` label.
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
                # Physics blow-ups (NaN states) end the episode in stock Kinetix but can leak a
                # NaN reward through the dense shaping; one NaN update kills the run for good.
                reward = jnp.nan_to_num(reward)

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
                        "train/success_rate": (raw_info["GoalR"] * dones).sum() / denom,
                        "train/episode_length": (raw_info["returned_episode_lengths"] * dones).sum() / denom,
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

    def _make_run_chunk(update_fn, num_updates):
        def run(runner_state):
            return jax.lax.scan(update_fn, runner_state, None, num_updates)

        return jax.jit(run)

    init_fns = {r: make_init_fn(envs[r]) for r in envs}
    update_fns = {r: make_update_step(envs[r], r) for r in envs}
    run_chunk_cache = {}

    def run_round(round_idx, runner_state, rng, start_done=0, on_chunk=None):
        """Run one round's updates in chunks of ``video_freq_updates``, logging
        GIFs of the fixed video episodes between chunks and calling ``on_chunk``
        (checkpointing) after every chunk. ``start_done`` resumes a partially
        completed round. Compiles once per distinct chunk length."""
        num_updates = int(config["num_updates_per_round"])
        chunk = int(config.get("video_freq_updates", 0)) or num_updates
        chunk = max(1, min(chunk, num_updates))
        done = int(start_done)
        while done < num_updates:
            n = min(chunk, num_updates - done)
            key = (round_idx, n)
            if key not in run_chunk_cache:
                run_chunk_cache[key] = _make_run_chunk(update_fns[round_idx], n)
            runner_state, _ = run_chunk_cache[key](runner_state)
            done += n
            if on_chunk is not None:
                on_chunk(round_idx, done, runner_state)
            if done < num_updates:
                rng, _rng = jax.random.split(rng)
                videos = jax.device_get(
                    video_eval_fn(_rng, runner_state.train_state, runner_state.extra["rms"], jnp.asarray(round_idx - 1))
                )
                _log_videos(videos, round_idx, runner_state.update_step)
        return runner_state

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

    ckpt_dir = Path(config["ckpt_dir"]) if config.get("ckpt_dir") else None

    def save_ckpt(round_idx, done_updates, runner_state, M, boundary_done, rng):
        """Atomic checkpoint (params, optimizer, RMS, boundary matrix, counters, W&B run id)
        so a preempted/requeued SLURM job resumes mid-round in the same W&B run."""
        if ckpt_dir is None or jax.process_index() != 0:
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(
            round=int(round_idx),
            done_updates=int(done_updates),
            boundary_done=bool(boundary_done),
            update_step=int(runner_state.update_step),
            train_state=jax.device_get(serialization.to_state_dict(runner_state.train_state)),
            extra=jax.device_get(runner_state.extra),
            M={int(b): {int(j): float(v) for j, v in row.items()} for b, row in M.items()},
            rng=np.asarray(jax.device_get(rng)),
            wandb_run_id=(wandb.run.id if (config["use_wandb"] and wandb.run is not None) else None),
        )
        tmp = ckpt_dir / "latest.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, ckpt_dir / "latest.pkl")
        logger.info(f"[CKPT] saved round={round_idx} done={done_updates}/{config['num_updates_per_round']} boundary_done={boundary_done}")

    def train(rng, resume=None):
        # INIT NETWORK (all rounds share obs/action spaces; use the round-1 env)
        rng, _rng = jax.random.split(rng)
        train_state = get_train_state_from_config(config, _rng, envs[1], env_params)

        if resume is not None:
            train_state = serialization.from_state_dict(train_state, resume["train_state"])
            M = {int(b): {int(j): float(v) for j, v in row.items()} for b, row in resume["M"].items()}
            start_round, start_done, boundary_done = int(resume["round"]), int(resume["done_updates"]), bool(resume["boundary_done"])
            update_step0 = int(resume["update_step"])
            rng = jnp.asarray(resume["rng"], dtype=jnp.uint32)
            logger.info(f"[RESUME] round={start_round} done={start_done} boundary_done={boundary_done} update_step={update_step0}")
        else:
            M, start_round, start_done, boundary_done, update_step0 = {}, 1, 0, True, 0

        rng, _rng = jax.random.split(rng)
        rngs_per_device = jax.device_put(jax.random.split(_rng, NUM_GPUS), partitioned_sharding)
        obsv, env_state, init_hstate, init_dones = init_fns[start_round](rngs_per_device)
        initial_extra = resume["extra"] if resume is not None else {"rms": rms_init(jax.tree.map(lambda x: x[0], obsv))}

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=jax.device_put(train_state, replicated_sharding),
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            extra=jax.device_put(initial_extra, replicated_sharding),
            hstate=init_hstate,
            rng=jax.random.split(_rng, NUM_GPUS),
            update_step=jax.device_put(jnp.array(update_step0), replicated_sharding),
        )

        # M[b][j] = success on pool j measured at the boundary after round b (b = 0: untrained).

        def _full_eval_and_log(b):
            nonlocal rng
            rng, _rng, _rng_v = jax.random.split(rng, 3)
            metrics = jax.device_get(full_eval_fn(_rng, runner_state.train_state, runner_state.extra["rms"]))
            M[b] = {j: float(metrics[f"eval/pool{j}_success"]) for j in range(1, num_rounds + 1)}
            extra = {f"boundary/S_pool{j}": M[b][j] for j in range(1, num_rounds + 1)}
            extra["boundary/after_round"] = b
            if b >= 1:
                extra[f"fig6/S_end_round"] = M[b][b]  # S_end(d_b), x-axis = round
                if b < num_rounds:
                    extra["fig6/S_start_next_round"] = M[b][b + 1]
                    extra["fig6/delta_next"] = M[b][b] - M[b][b + 1]  # Δ_{b+1}
            _log_full_eval(metrics, extra, runner_state.update_step, round_idx=max(b, 1))
            videos = jax.device_get(
                video_eval_fn(_rng_v, runner_state.train_state, runner_state.extra["rms"], jnp.asarray(max(b, 1) - 1))
            )
            _log_videos(videos, max(b, 1), runner_state.update_step)

        def _on_chunk(round_idx, done, rs):
            save_ckpt(round_idx, done, rs, M, boundary_done=False, rng=rng)

        if resume is None:
            _full_eval_and_log(0)
            save_ckpt(1, 0, runner_state, M, boundary_done=True, rng=rng)

        for r in range(start_round, num_rounds + 1):
            if r > start_round:
                # Fresh envs from pool r; params, optimizer state, RMS state and the
                # global update counter carry over unchanged.
                rng, _rng, _rng2 = jax.random.split(rng, 3)
                rngs_per_device = jax.device_put(jax.random.split(_rng, NUM_GPUS), partitioned_sharding)
                obsv_r, env_state_r, init_hstate_r, init_dones_r = init_fns[r](rngs_per_device)
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state_r,
                    last_obs=obsv_r,
                    last_done=init_dones_r,
                    extra=runner_state.extra,
                    hstate=init_hstate_r,
                    rng=jax.random.split(_rng2, NUM_GPUS),
                    update_step=runner_state.update_step,
                )
            done0 = start_done if r == start_round else 0
            if done0 < config["num_updates_per_round"]:
                logger.info(f"Round {r}/{num_rounds}: updates {done0}->{config['num_updates_per_round']} on pool {r}")
                rng, _rng = jax.random.split(rng)
                runner_state = run_round(r, runner_state, _rng, start_done=done0, on_chunk=_on_chunk)
                jax.block_until_ready(runner_state.train_state.params)
            assert int(runner_state.update_step) == r * config["num_updates_per_round"], (
                int(runner_state.update_step), r, config["num_updates_per_round"])
            if not (r == start_round and done0 >= config["num_updates_per_round"] and boundary_done):
                _full_eval_and_log(r)
                save_ckpt(r, config["num_updates_per_round"], runner_state, M, boundary_done=True, rng=rng)

        # Transfer summary from the boundary matrix.
        R = num_rounds
        transfer = {}
        for r in range(2, R + 1):
            transfer[f"transfer/delta_{r}"] = M[r - 1][r - 1] - M[r - 1][r]
        for j in range(1, R):
            transfer[f"transfer/B_{R}_{j}"] = M[R][j] - M[j][j]
        for j in range(1, R + 1):
            transfer[f"final/S_end_pool{j}"] = M[j][j]
            transfer[f"final/S_final_pool{j}"] = M[R][j]
        deltas = [transfer[f"transfer/delta_{r}"] for r in range(2, R + 1)]
        transfer["transfer/delta_mean"] = float(np.mean(deltas)) if deltas else 0.0
        transfer["transfer/S_end_mean"] = float(np.mean([M[j][j] for j in range(1, R + 1)]))
        transfer["transfer/S_final_mean"] = float(np.mean([M[R][j] for j in range(1, R + 1)]))
        if R > 1:
            transfer["transfer/B_R_1"] = transfer[f"transfer/B_{R}_1"]
            transfer["transfer/delta_2"] = transfer["transfer/delta_2"]
        transfer["round"] = R
        transfer["timing/num_env_steps"] = _global_env_steps(runner_state.update_step)
        logger.info(f"Transfer summary: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in transfer.items()} }")
        if config["use_wandb"] and jax.process_index() == 0:
            wandb.log(transfer)
            wandb.run.summary["boundary_matrix"] = {str(b): {str(j): v for j, v in row.items()} for b, row in M.items()}

        if config["save_policy"] and jax.process_index() == 0:
            save_model(
                runner_state.train_state,
                _global_env_steps(runner_state.update_step),
                config,
                is_final=True,
                save_to_wandb=config["use_wandb"],
                extra=jax.device_get(runner_state.extra),
            )
        return {"runner_state": runner_state, "boundary_matrix": M, "transfer": transfer}

    return train


@hydra.main(version_base=None, config_path="../configs", config_name="ppo_rounds")
def main(config):
    name = "PPO-Rounds"
    config = OmegaConf.to_container(config)
    config["learning"]["total_timesteps"] = int(config["num_rounds"]) * int(config["steps_per_round"])
    config = normalise_config(config, name)

    env_params, static_env_params = generate_params_from_config(config)
    if config.get("max_timesteps") is not None:
        env_params = env_params.replace(max_timesteps=int(config["max_timesteps"]))
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    config["kinetix_version"] = "v3.0.0"
    try:
        config["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        config["git_sha"] = "unknown"

    steps_per_update = int(config["num_steps"]) * int(config["num_train_envs"])
    config["num_updates_per_round"] = int(config["steps_per_round"]) // steps_per_update
    assert config["num_updates_per_round"] > 0
    num_rounds = int(config["num_rounds"])
    config["diversity/tasks_per_round"] = int(config["tasks_per_round"])
    logger.info(
        f"[POOLS] rounds={num_rounds} tasks_per_round={config['tasks_per_round']} "
        f"steps_per_round={config['steps_per_round']} updates_per_round={config['num_updates_per_round']}"
    )

    envs = {
        r: make_env(config, static_env_params, env_params, make_pool_reset_fn(config, env_params, static_env_params, r))
        for r in range(1, num_rounds + 1)
    }

    assert config["eval_num_episodes_periodic"] <= config["eval_num_episodes_full"]
    assert config["eval_num_video_episodes"] <= config["eval_num_episodes_full"]
    full_pools = {
        r: build_pool_levels(config, env_params, static_env_params, r, config["eval_num_episodes_full"])
        for r in range(1, num_rounds + 1)
    }
    periodic_pools = {r: jax.tree.map(lambda x: x[: config["eval_num_episodes_periodic"]], v) for r, v in full_pools.items()}
    video_pools = {r: jax.tree.map(lambda x: x[: config["eval_num_video_episodes"]], v) for r, v in full_pools.items()}

    # Checkpoint/resume (for preemptible partitions): one directory per (group, run name, seed).
    if not config.get("ckpt_dir"):
        tag = (config.get("extra_run_name") or "run").strip("_") or "run"
        config["ckpt_dir"] = str(REPO_ROOT / "checkpoints" / "rounds" / str(config["group"]) / f"{tag}_s{config['seed']}")
    resume = None
    latest = Path(config["ckpt_dir"]) / "latest.pkl"
    if latest.exists() and not config.get("no_resume", False):
        with latest.open("rb") as f:
            resume = pickle.load(f)
        logger.info(f"[RESUME] found checkpoint {latest}: round={resume['round']} done={resume['done_updates']} wandb={resume.get('wandb_run_id')}")

    if config["use_wandb"]:
        if jax.process_index() == 0:
            kw = dict(settings=wandb.Settings(quiet=True))
            if resume is not None and resume.get("wandb_run_id"):
                kw |= dict(id=resume["wandb_run_id"], resume="allow")
            init_wandb(config, name, **kw)
        else:
            os.environ["WANDB_MODE"] = "disabled"

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    train = make_train(config, env_params, static_env_params, envs, full_pools, periodic_pools, video_pools)
    train(_rng, resume=resume)


if __name__ == "__main__":
    main()
