"""Phase-2 gate: is the Banyan task physically solvable by the Kinetix claw?

A CEM planner (a NON-LEARNING controller) searches open-loop piecewise-constant
motor sequences in the real physics. The pass metric is the environment's real
sparse reward — never the planner's internal shaping.

depth 1: one stage — deliver the goal-typed object into the combine zone.
depth 2: two stages — deliver required object A, then plan onward from the
reached state to deliver required object B (merge fires on co-occupancy).

Usage:
  python -m kinetix_banyan.gate_cem --depth 1 --tasks 8 --episodes 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BANYAN_GRID = REPO.parent / "banyan-grid"
for p in (REPO, BANYAN_GRID):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark.baselines.utils.continuous_banyan import build_d1_d2_task_banks

from kinetix_banyan.task_layer import (
    OBJECT_SLOTS,
    make_banyan_env,
    make_banyan_reset_fn,
    prepare_banyan_level,
)

# Claw geometry: far finger segments of grasp_easy.
FINGER_POLYS = (8, 9)


def build_setup(num_instances: int, seed: int, id_vocab: int = 120):
    d1_bank, _d2_bank, meta = build_d1_d2_task_banks(
        d1_num_instances=num_instances,
        d2_num_instances=num_instances,
        num_leaf_slots=len(OBJECT_SLOTS),
        id_vocab_size=id_vocab,
        recipe_length=2,
        seed=seed,
        task_bank_mode="banyan_items_curriculum",
        object_condition="shared",
    )
    rb = meta["global_rulebook"]
    base, static, ep, consts = prepare_banyan_level(
        merge_lhs=np.asarray(rb["merge_lhs"]),
        merge_rhs=np.asarray(rb["merge_rhs"]),
        token_vocab_size=int(meta["token_vocab_size"]),
    )
    reset_fn = make_banyan_reset_fn(base, consts, d1_bank)
    env = make_banyan_env(
        base_state=base,
        static_env_params=static,
        env_params=ep,
        constants=consts,
        reset_fn=reset_fn,
        action_type="continuous",
        auto_reset=False,
    )
    return env, ep, consts, d1_bank, base


def make_planner(env, ep, consts, *, horizon, n_segments, pop, elites, iters, action_dim):
    """CEM over piecewise-constant motor actions. Returns plan(state, target_type)."""
    assert horizon % n_segments == 0, (
        f"horizon {horizon} must be divisible by n_segments {n_segments}"
    )
    seg_len = horizon // n_segments
    zone_c = 0.5 * (consts.zone_lo + consts.zone_hi)
    obj_idx = jnp.asarray(OBJECT_SLOTS)
    fingers = jnp.asarray(FINGER_POLYS)

    def rollout(state0, segments, target_type):
        """segments: (n_segments, action_dim). Returns (score, real_return, steps_to_done)."""
        acts = jnp.repeat(segments, seg_len, axis=0)  # (horizon, action_dim)
        acts = jnp.concatenate(
            [acts, jnp.zeros((horizon, 2))], axis=1
        )  # thruster dims (inactive in level)

        def step(carry, a):
            st, done_prev, ret, steps = carry
            _o, st1, r, done, _i = env.step_env(jax.random.PRNGKey(0), st, a, ep)
            st = jax.tree.map(lambda x, y: jax.lax.select(done_prev, x, y), st, st1)
            r = jnp.where(done_prev, 0.0, r)
            ret = ret + r
            steps = steps + jnp.where(done_prev | done, 0, 1)
            # planner guidance (never the gate metric):
            is_target = st.circle_types[obj_idx] == target_type
            opos = st.circle.position[obj_idx]
            d_obj_zone = jnp.min(
                jnp.where(is_target, jnp.linalg.norm(opos - zone_c[None], axis=1), 1e6)
            )
            tip = st.polygon.position[fingers].mean(axis=0)
            d_tip_obj = jnp.min(
                jnp.where(is_target, jnp.linalg.norm(opos - tip[None], axis=1), 1e6)
            )
            return (st, done_prev | done, ret, steps), (d_obj_zone, d_tip_obj)

        (stT, _, ret, steps), (d_zone_t, d_tip_t) = jax.lax.scan(
            step, (state0, jnp.asarray(False), 0.0, 0), acts
        )
        score = (
            1000.0 * ret
            - 3.0 * d_zone_t[-1]
            - 1.0 * jnp.min(d_zone_t)
            - 1.0 * jnp.min(d_tip_t)
            - 0.3 * d_tip_t[-1]
        )
        return score, ret, steps, stT

    v_rollout = jax.vmap(rollout, in_axes=(None, 0, None))

    @jax.jit
    def plan(rng, state0, target_type):
        mu = jnp.zeros((n_segments, action_dim))
        sigma = jnp.ones((n_segments, action_dim)) * 0.8

        def cem_iter(carry, _):
            rng, mu, sigma = carry
            rng, k = jax.random.split(rng)
            cand = mu[None] + sigma[None] * jax.random.normal(k, (pop, n_segments, action_dim))
            cand = jnp.clip(cand, -1.0, 1.0)
            scores, rets, steps, _ = v_rollout(state0, cand, target_type)
            elite_idx = jnp.argsort(-scores)[:elites]
            elite = cand[elite_idx]
            mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
            sigma = 0.7 * elite.std(axis=0) + 0.3 * sigma + 0.02
            best = elite_idx[0]
            return (rng, mu, sigma), (scores[best], rets[best], cand[best])

        (rng, mu, sigma), (bscores, brets, bcands) = jax.lax.scan(
            cem_iter, (rng, mu, sigma), None, length=iters
        )
        overall = jnp.argmax(bscores)
        best_seg = bcands[overall]
        score, ret, steps, stT = rollout(state0, best_seg, target_type)
        return best_seg, ret, steps, stT

    return plan, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=1, choices=[1, 2])
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=192, help="steps per stage")
    ap.add_argument("--segments", type=int, default=24)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--elites", type=int, default=32)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Override env max_timesteps (horizon escalation — report loudly).",
    )
    args = ap.parse_args()

    env, ep, consts, d1_bank, base = build_setup(args.instances, args.seed)
    if args.max_timesteps is not None:
        ep = ep.replace(max_timesteps=int(args.max_timesteps))
        print(f"[HORIZON OVERRIDE] max_timesteps -> {args.max_timesteps}")
    plan, _ = make_planner(
        env,
        ep,
        consts,
        horizon=args.horizon,
        n_segments=args.segments,
        pop=args.pop,
        elites=args.elites,
        iters=args.iters,
        action_dim=4,
    )

    depths = np.asarray(d1_bank["depth"])
    task_rows = np.nonzero(depths == args.depth)[0][: args.tasks]
    rows_out = []
    t_start = time.time()
    for task_i, row in enumerate(task_rows):
        row_bank = {k: v[int(row) : int(row) + 1] for k, v in d1_bank.items()}
        from kinetix_banyan.task_layer import make_banyan_reset_fn as _mk

        rf = _mk(base, consts, row_bank)
        for epi in range(args.episodes):
            rng = jax.random.PRNGKey(args.seed * 7919 + task_i * 101 + epi)
            state = rf(rng)
            goal = int(state.goal_token)
            lhs, rhs = int(state.required_lhs), int(state.required_rhs)
            t0 = time.time()
            if args.depth == 1:
                _seg, ret, steps, stT = plan(rng, state, jnp.asarray(goal))
                total_ret = float(ret)
                total_steps = int(steps)
            else:
                # stage A: deliver lhs; stage B: continue, deliver rhs.
                _segA, retA, stepsA, stA = plan(rng, state, jnp.asarray(lhs))
                rngB = jax.random.fold_in(rng, 1)
                _segB, retB, stepsB, stT = plan(rngB, stA, jnp.asarray(rhs))
                total_ret = float(retA) + float(retB)
                total_steps = int(stepsA) + int(stepsB)
            success = total_ret > 0.5
            deadend = total_ret < -0.5
            rows_out.append(
                dict(
                    task=int(row),
                    episode=epi,
                    depth=args.depth,
                    success=bool(success),
                    deadend=bool(deadend),
                    steps=total_steps,
                    ret=total_ret,
                    wallclock_s=round(time.time() - t0, 1),
                )
            )
            print(f"[task {task_i} ep {epi}] success={success} deadend={deadend} steps={total_steps} ret={total_ret:+.1f} ({rows_out[-1]['wallclock_s']}s)", flush=True)

    n = len(rows_out)
    succ = sum(r["success"] for r in rows_out)
    dead = sum(r["deadend"] for r in rows_out)
    steps_ok = [r["steps"] for r in rows_out if r["success"]]
    summary = dict(
        depth=args.depth,
        episodes=n,
        solve_rate=succ / n if n else float("nan"),
        deadend_rate=dead / n if n else float("nan"),
        median_steps_success=float(np.median(steps_ok)) if steps_ok else float("nan"),
        p95_steps_success=float(np.percentile(steps_ok, 95)) if steps_ok else float("nan"),
        stage_horizon=args.horizon,
        max_timesteps_env=int(ep.max_timesteps),
        planner=dict(pop=args.pop, iters=args.iters, segments=args.segments, elites=args.elites),
        wallclock_total_s=round(time.time() - t_start, 1),
    )
    print(json.dumps(summary, indent=2))
    out = args.out or Path(f"outputs/gate/gate_depth{args.depth}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=rows_out), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
