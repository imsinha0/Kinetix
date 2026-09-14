"""Figure-6-style plots for the Kinetix locomotion rounds experiment.

  python -m kinetix_rounds.plot_fig6 --group loco-r10-v1 [--no-wandb]

Panels (one line/bar per diversity n = tasks_per_round, mean over seeds, min–max band):
  fig6_S_end_vs_round.png   S_end(d_r) = success on pool r right after round r, vs round r
  fig6_success_vs_steps.png current-pool success and pool-1 success vs env steps (round boundaries dashed)
  fig6_delta_vs_round.png   forward gap Δ_r = S_end(d_{r-1}) − S_start(d_r), vs round
  fig6_B_bars.png           pool-1 success after round 1 (blue) vs after the last round (red) = B(R,1) components
Uses finished runs only (transfer summary present); running runs are ignored.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENTITY, PROJECT = "imsinha-harvard-university", "kinetix-rounds"
PRELIM = "PRELIMINARY — Kinetix substrate, few seeds. Pattern only. Not for publication."


def fetch(group, include_running=False):
    api = wandb.Api()
    out = []
    for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group}):
        s = dict(r.summary)
        if not include_running and (r.state != "finished" or "transfer/S_end_mean" not in s):
            continue
        cfg = r.config
        R = int(cfg.get("num_rounds", 10))
        n = int(cfg.get("tasks_per_round", -1))
        M = s.get("boundary_matrix")
        if M is None:
            # reconstruct from the boundary history (boundary/after_round + boundary/S_pool{j})
            M = {}
            for row in r.history(keys=["boundary/after_round"] + [f"boundary/S_pool{j}" for j in range(1, R + 1)], pandas=False):
                if row.get("boundary/after_round") is None:
                    continue
                b = int(row["boundary/after_round"])
                M[str(b)] = {str(j): row.get(f"boundary/S_pool{j}") for j in range(1, R + 1)}
        keys = ["timing/num_env_steps", "round", "eval/pool1_success"] + [f"eval/pool{j}_success" for j in range(1, R + 1)]
        hist = [x for x in r.history(keys=keys, pandas=False) if x.get("eval/pool1_success") is not None]
        out.append(dict(run=r, n=n, R=R, seed=int(cfg.get("seed", 0)), M=M, hist=hist, summary=s))
    out.sort(key=lambda d: (d["n"], d["seed"]))
    return out


def _agg(vals):
    vals = np.asarray(vals, dtype=float)
    return np.nanmean(vals, axis=0), np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="loco-r10-v1")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--include-running", action="store_true", help="also use unfinished runs (boundaries so far)")
    args = ap.parse_args()
    outdir = args.outdir or Path("outputs/rounds") / args.group
    outdir.mkdir(parents=True, exist_ok=True)
    runs = fetch(args.group, args.include_running)
    assert runs, f"no usable runs in group {args.group}"
    ns = sorted(set(d["n"] for d in runs))
    R = max(d["R"] for d in runs)
    cmap = plt.get_cmap("viridis")
    color = {n: cmap(i / max(1, len(ns) - 1)) for i, n in enumerate(ns)}
    label = {n: f"n={n} ({sum(d['n'] == n for d in runs)} seeds)" for n in ns}

    def S_end(d, r):  # M[r][r]
        return float(d["M"].get(str(r), {}).get(str(r), np.nan))

    def S_start(d, r):  # M[r-1][r]
        return float(d["M"].get(str(r - 1), {}).get(str(r), np.nan))

    rows = []
    # ---- Panel: S_end(d_r) vs r ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n in ns:
        curves = [[S_end(d, r) for r in range(1, R + 1)] for d in runs if d["n"] == n]
        m, lo, hi = _agg(curves)
        x = np.arange(1, R + 1)
        ax.plot(x, m, "o-", color=color[n], label=label[n])
        ax.fill_between(x, lo, hi, color=color[n], alpha=0.15)
    ax.set_xlabel("round r (each round = a disjoint pool of n levels)")
    ax.set_ylabel("S_end(d_r): success on pool r after training on it")
    ax.set_xticks(range(1, R + 1))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Success reached on each round's tasks [{args.group}]\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p1 = outdir / "fig6_S_end_vs_round.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- Panel: success vs env steps ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    steps_per_round = None
    for n in ns:
        grid_curves_cur, grid_curves_p1, grid = [], [], None
        for d in runs:
            if d["n"] != n or not d["hist"]:
                continue
            h = d["hist"]
            steps = np.array([x["timing/num_env_steps"] for x in h], float)
            rounds = np.array([int(x.get("round", 1) or 1) for x in h])
            cur = np.array([x.get(f"eval/pool{rr}_success", np.nan) for x, rr in zip(h, rounds)], float)
            p1s = np.array([x["eval/pool1_success"] for x in h], float)
            axes[0].plot(steps / 1e6, cur, color=color[n], alpha=0.2, lw=0.8)
            axes[1].plot(steps / 1e6, p1s, color=color[n], alpha=0.2, lw=0.8)
            if grid is None:
                grid = np.linspace(0, steps.max(), 400)
            o = np.argsort(steps)
            grid_curves_cur.append(np.interp(grid, steps[o], cur[o]))
            grid_curves_p1.append(np.interp(grid, steps[o], p1s[o]))
            steps_per_round = float(d["run"].config.get("steps_per_round", 1e8))
        if grid is not None:
            axes[0].plot(grid / 1e6, np.nanmean(grid_curves_cur, axis=0), color=color[n], lw=2.2, label=label[n])
            axes[1].plot(grid / 1e6, np.nanmean(grid_curves_p1, axis=0), color=color[n], lw=2.2, label=label[n])
    for ax, title in zip(axes, ["success on the current round's pool", "success on pool 1 (round-1 tasks) throughout"]):
        if steps_per_round:
            for r in range(1, R):
                ax.axvline(r * steps_per_round / 1e6, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("env steps (M)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Kinetix, {R} rounds of 100M steps on disjoint task pools [{args.group}]\n{PRELIM}", fontsize=9)
    fig.tight_layout()
    p2 = outdir / "fig6_success_vs_steps.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- Panel: Δ_r vs r ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n in ns:
        curves = [[S_end(d, r - 1) - S_start(d, r) for r in range(2, R + 1)] for d in runs if d["n"] == n]
        m, lo, hi = _agg(curves)
        x = np.arange(2, R + 1)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", capsize=3, color=color[n], label=label[n])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("round r")
    ax.set_ylabel("Δ_r = S_end(d_{r-1}) − S_start(d_r)")
    ax.set_xticks(range(2, R + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Forward-transfer gap at each round boundary [{args.group}]\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p3 = outdir / "fig6_delta_vs_round.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ---- Panel: B(R,1) bars ----
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(ns))
    w = 0.38
    a1 = [[S_end(d, 1) for d in runs if d["n"] == n] for n in ns]
    a2 = [[float(d["M"].get(str(R), {}).get("1", np.nan)) for d in runs if d["n"] == n] for n in ns]
    m1 = [np.nanmean(v) for v in a1]; m2 = [np.nanmean(v) for v in a2]
    ax.bar(x - w / 2, m1, w, color="#6688cc", label="pool 1 after round 1 (S_end(d1))")
    ax.bar(x + w / 2, m2, w, color="#cc6655", label=f"pool 1 after round {R} (S_final(d1))")
    for xi, v1, v2 in zip(x, a1, a2):
        ax.scatter([xi - w / 2] * len(v1), v1, s=10, color="k", alpha=0.6, zorder=3)
        ax.scatter([xi + w / 2] * len(v2), v2, s=10, color="k", alpha=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n = tasks per round")
    ax.set_ylabel("success on pool-1 tasks")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"B({R},1) components (red > blue = positive backward transfer) [{args.group}]\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p4 = outdir / "fig6_B_bars.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    # ---- CSV ----
    csv_path = outdir / "fig6_metrics.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["n", "seed", "run_id"] + [f"S_end_r{r}" for r in range(1, R + 1)] + [f"delta_r{r}" for r in range(2, R + 1)] + [f"B_{R}_{j}" for j in range(1, R)])
        for d in runs:
            wri.writerow(
                [d["n"], d["seed"], d["run"].id]
                + [S_end(d, r) for r in range(1, R + 1)]
                + [S_end(d, r - 1) - S_start(d, r) for r in range(2, R + 1)]
                + [float(d["M"].get(str(R), {}).get(str(j), np.nan)) - S_end(d, j) for j in range(1, R)]
            )
    print(f"wrote {p1}, {p2}, {p3}, {p4}, {csv_path}")
    for n in ns:
        m, _, _ = _agg([[S_end(d, r) for r in range(1, R + 1)] for d in runs if d["n"] == n])
        print(f"n={n}: S_end by round = {np.round(m, 3).tolist()}")
    if args.no_wandb:
        return
    run = wandb.init(entity=ENTITY, project=PROJECT, group=args.group, name=f"Figure 6 — locomotion rounds ({args.group})", job_type="analysis", notes=PRELIM)
    run.log({f"figures/{p.stem}": wandb.Image(str(p)) for p in (p1, p2, p3, p4)})
    art = wandb.Artifact("fig6_data", type="analysis")
    for p in (p1, p2, p3, p4, csv_path):
        art.add_file(str(p))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
