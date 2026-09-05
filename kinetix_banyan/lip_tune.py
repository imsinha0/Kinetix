"""Lip-height tuning: null-policy battery at a candidate lip height; if the
type-blind floor stays <= threshold, run the CEM gate (depths 1 and 2) at
that height. One GPU job, one JSON verdict."""

from __future__ import annotations

import argparse
import json
import sys
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

from kinetix_banyan.gate_cem import make_planner
from kinetix_banyan.task_layer import (
    OBJECT_SLOTS,
    make_banyan_env,
    make_banyan_reset_fn,
    prepare_banyan_level,
)

CONSTS = [
    [-1.0, -1.0, -1.0, -1.0],
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
]


def null_battery(env, ep, rf, n_eps: int, horizon: int) -> dict:
    def make_ep(mode, const=None):
        def episode(rng):
            rng_r, rng_a = jax.random.split(rng)
            st = rf(rng_r)

            def step(carry, t):
                st, done_prev, ret = carry
                if mode == "const":
                    a = jnp.asarray(const + [0.0, 0.0])
                else:
                    a = jax.random.uniform(
                        jax.random.fold_in(rng_a, t), (6,), minval=-1.0, maxval=1.0
                    )
                _o, st1, r, done, _i = env.step_env(jax.random.PRNGKey(0), st, a, ep)
                st = jax.tree.map(lambda x, y: jax.lax.select(done_prev, x, y), st, st1)
                r = jnp.where(done_prev, 0.0, r)
                return (st, done_prev | done, ret + r), None

            (stT, done, ret), _ = jax.lax.scan(
                step, (st, jnp.asarray(False), 0.0), jnp.arange(horizon)
            )
            return ret

        return episode

    out = {}
    for const in CONSTS:
        rets = np.asarray(
            jax.jit(jax.vmap(make_ep("const", const)))(
                jax.random.split(jax.random.PRNGKey(1), n_eps)
            )
        )
        out[f"const_{const}"] = float(np.mean(rets > 0.5))
    rets = np.asarray(
        jax.jit(jax.vmap(make_ep("rand")))(jax.random.split(jax.random.PRNGKey(2), n_eps))
    )
    out["random"] = float(np.mean(rets > 0.5))
    out["worst"] = max(v for k, v in out.items() if k != "random")
    return out


def run_gate(env, ep, consts, d1_bank, base, depth, tasks, episodes, pop, iters, horizon, segments):
    plan, _ = make_planner(
        env, ep, consts, horizon=horizon, n_segments=segments, pop=pop, elites=32, iters=iters, action_dim=4
    )
    depths = np.asarray(d1_bank["depth"])
    task_rows = np.nonzero(depths == depth)[0][:tasks]
    rows = []
    for ti, row in enumerate(task_rows):
        row_bank = {k: v[int(row) : int(row) + 1] for k, v in d1_bank.items()}
        rf = make_banyan_reset_fn(base, consts, row_bank)
        for epi in range(episodes):
            rng = jax.random.PRNGKey(7919 + ti * 101 + epi)
            state = rf(rng)
            lhs, rhs = int(state.required_lhs), int(state.required_rhs)
            goal = int(state.goal_token)
            if depth == 1:
                _s, ret, steps, _ = plan(rng, state, jnp.asarray(goal))
                total_ret, total_steps = float(ret), int(steps)
            else:
                _s, retA, stepsA, stA = plan(rng, state, jnp.asarray(lhs))
                _s, retB, stepsB, _ = plan(jax.random.fold_in(rng, 1), stA, jnp.asarray(rhs))
                total_ret, total_steps = float(retA) + float(retB), int(stepsA) + int(stepsB)
            rows.append(dict(task=int(row), episode=epi, success=total_ret > 0.5, steps=total_steps))
            print(f"  gate d{depth} task {ti} ep {epi}: success={total_ret > 0.5} steps={total_steps}", flush=True)
    succ = [r for r in rows if r["success"]]
    return dict(
        solve_rate=len(succ) / len(rows),
        median_steps=float(np.median([r["steps"] for r in succ])) if succ else float("nan"),
        episodes=len(rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lip", type=float, default=0.40)
    ap.add_argument("--floor-threshold", type=float, default=0.02)
    ap.add_argument("--max-timesteps", type=int, default=512)
    ap.add_argument("--battery-eps", type=int, default=128)
    ap.add_argument("--gate-tasks", type=int, default=8)
    ap.add_argument("--gate-episodes", type=int, default=2)
    ap.add_argument("--pop", type=int, default=1536)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("outputs/gate/lip_tune.json"))
    args = ap.parse_args()

    d1, _d2, meta = build_d1_d2_task_banks(
        d1_num_instances=8, d2_num_instances=8, num_leaf_slots=4, id_vocab_size=120,
        recipe_length=2, seed=0, task_bank_mode="banyan_items_curriculum", object_condition="shared",
    )
    rb = meta["global_rulebook"]
    base, static, ep, consts = prepare_banyan_level(
        merge_lhs=np.asarray(rb["merge_lhs"]), merge_rhs=np.asarray(rb["merge_rhs"]),
        token_vocab_size=int(meta["token_vocab_size"]), lip_height=args.lip,
    )
    ep = ep.replace(max_timesteps=args.max_timesteps)
    rf = make_banyan_reset_fn(base, consts, d1)
    env = make_banyan_env(
        base_state=base, static_env_params=static, env_params=ep, constants=consts,
        reset_fn=rf, action_type="continuous", auto_reset=False,
    )

    print(f"[lip={args.lip}] null battery ({args.battery_eps} eps x {args.max_timesteps} steps)...", flush=True)
    battery = null_battery(env, ep, rf, args.battery_eps, args.max_timesteps)
    print(json.dumps(battery, indent=2), flush=True)
    result = dict(lip=args.lip, max_timesteps=args.max_timesteps, battery=battery)

    if battery["worst"] <= args.floor_threshold and battery["random"] <= args.floor_threshold:
        print(f"[lip={args.lip}] battery PASS (worst {battery['worst']:.3f}) -> gating", flush=True)
        result["gate_d1"] = run_gate(env, ep, consts, d1, base, 1, args.gate_tasks, args.gate_episodes,
                                     1024, 28, 256, 32)
        result["gate_d2"] = run_gate(env, ep, consts, d1, base, 2, args.gate_tasks, args.gate_episodes,
                                     args.pop, args.iters, 256, 32)
        result["verdict"] = "PASS" if result["gate_d2"]["solve_rate"] >= 0.6 else "GATE_LOW"
    else:
        result["verdict"] = "FLOOR_TOO_HIGH"
        print(f"[lip={args.lip}] battery FAIL (worst {battery['worst']:.3f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "battery"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
