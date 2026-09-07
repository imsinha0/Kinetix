"""Recreate Figure 5 for the Kinetix substrate from a sweep group on W&B.

Pulls the finished runs of --group (one per |O|; --filter selects a subset by
run-name substring, e.g. "nd0"), extracts eval histories + transfer metrics for
the configured task depths, renders the panels, and logs them back to W&B.

  python -m kinetix_banyan.plot_figure5 --group d1-v1 --filter nd0 --depths 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "imsinha-harvard-university"
PROJECT = "kinetix-banyan"
PRELIM = "PRELIMINARY — Kinetix substrate, few seeds. Pattern only. Not for publication."


def fetch_runs(group: str, depths: tuple[int, ...], name_filter: str = ""):
    api = wandb.Api()
    runs = [
        r
        for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
        if r.state == "finished" and name_filter in r.name and "transfer/delta_2" in r.summary
    ]
    out = []
    for r in runs:
        cfg = {k: v for k, v in r.config.items()}
        n_d1 = int(cfg.get("d1_num_instances", -1))
        summ = dict(r.summary)
        if float(summ.get("boundary/S_end_d1", 0.0) or 0.0) < 0.05:
            print(f"skipping dead run {r.name} ({r.id}): S_end_d1={summ.get('boundary/S_end_d1')}")
            continue
        succ_keys = [f"eval/{b}_success_depth{d}" for b in ("d1", "d2") for d in depths]
        hist = r.history(keys=["timing/num_env_steps", "round", *succ_keys], pandas=False)
        diag = (
            r.history(
                keys=["timing/num_env_steps", "eval/d1_deadend_depth2", "eval/d2_deadend_depth2"],
                pandas=False,
            )
            if 2 in depths
            else []
        )
        sel = r.history(
            keys=["timing/num_env_steps", "train/n_in_zone", "train/n_required_in_zone"],
            pandas=False,
        )
        out.append(
            dict(
                run=r,
                n_d1=n_d1,
                summary=summ,
                history=list(hist),
                diag=list(diag),
                sel=list(sel),
                floor=float(cfg.get("null_policy_floor", -1.0)),
                seed=int(cfg.get("seed", 0)),
            )
        )
    out.sort(key=lambda d: (d["n_d1"], d["seed"]))
    return out


def _agg(ns, vals):
    """Group values by |O|: returns sorted unique ns, means, mins, maxs."""
    uniq = sorted(set(ns))
    groups = [[v for n, v in zip(ns, vals) if n == u and np.isfinite(v)] for u in uniq]
    mean = np.array([np.mean(g) if g else np.nan for g in groups])
    lo = np.array([np.min(g) if g else np.nan for g in groups])
    hi = np.array([np.max(g) if g else np.nan for g in groups])
    return np.array(uniq), mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="d1-v1")
    ap.add_argument("--filter", default="", help="run-name substring, e.g. nd0")
    ap.add_argument("--depths", type=int, nargs="+", default=[1])
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    depths = tuple(args.depths)
    tag = f"{args.group}{'_' + args.filter if args.filter else ''}"
    args.outdir = args.outdir or Path("outputs/fig5") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = fetch_runs(args.group, depths, args.filter)
    assert runs, f"no finished runs in group {args.group} matching {args.filter!r}"
    print(f"found {len(runs)} runs: |O| = {[d['n_d1'] for d in runs]}")

    def _summ(d, key):
        return float(d["summary"].get(key, np.nan))

    ns = [d["n_d1"] for d in runs]
    delta2 = [_summ(d, "transfer/delta_2") for d in runs]
    delta2_d1 = [_summ(d, "transfer/delta_2_depth1") for d in runs]
    delta2_d2 = [_summ(d, "transfer/delta_2_depth2") for d in runs]
    # B(2,1) is meaningless for runs that died (NaN) during round 2; Δ₂ (boundary) is still valid.
    def _b21(d):
        dead_r2 = _summ(d, "final/S_end_d2") < 0.05 and _summ(d, "final/S_end_final_d1") < 0.05
        return np.nan if dead_r2 else _summ(d, "transfer/B_2_1")

    b21 = [_b21(d) for d in runs]
    b21_d1 = [_summ(d, "transfer/B_2_1_depth1") for d in runs]
    b21_d2 = [_summ(d, "transfer/B_2_1_depth2") for d in runs]

    def _mean_succ(row, bank):
        return float(np.mean([row[f"eval/{bank}_success_depth{dd}"] for dd in depths]))
    overlaps = [int(d["summary"].get("diversity/d1_d2_overlap", d["run"].config.get("diversity/d1_d2_overlap", -1))) for d in runs]
    assert all(o == 0 for o in overlaps), f"d1/d2 overlap nonzero: {overlaps}"

    # ---------- Panel 1: success vs env steps, one line per |O| ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    uniq_ns = sorted(set(ns))
    color_of = {n: cmap(i / max(1, len(uniq_ns) - 1)) for i, n in enumerate(uniq_ns)}
    # Individual seeds faint; per-|O| mean (on a common step grid) as the solid, labelled line.
    grid = np.linspace(0.0, max(float(row["timing/num_env_steps"]) for d in runs for row in d["history"]), 200)
    per_n = {n: {"cur": [], "d2": []} for n in uniq_ns}
    for d in runs:
        color = color_of[d["n_d1"]]
        h = d["history"]
        steps = np.array([row["timing/num_env_steps"] for row in h], dtype=float)
        # During round 1 the "current bank" is d1; during round 2 it's d2.
        cur = np.array([_mean_succ(row, "d1" if row.get("round", 1) == 1 else "d2") for row in h], dtype=float)
        d2s = np.array([_mean_succ(row, "d2") for row in h], dtype=float)
        # A run that died (NaN) during round 2 contributes only its round-1 segment to the means.
        dead_r2 = _summ(d, "final/S_end_d2") < 0.05 and _summ(d, "final/S_end_final_d1") < 0.05
        if dead_r2:
            r2 = np.array([row.get("round", 1) for row in h]) == 2
            cur[r2] = np.nan
            d2s[r2] = np.nan
        axes[0].plot(steps / 1e6, cur, color=color, alpha=0.2, lw=0.8)
        axes[1].plot(steps / 1e6, d2s, color=color, alpha=0.2, lw=0.8)
        order = np.argsort(steps)
        ok = ~np.isnan(cur[order])
        gcur = np.interp(grid, steps[order][ok], cur[order][ok])
        gd2 = np.interp(grid, steps[order][ok], d2s[order][ok])
        if dead_r2:
            gcur[grid > steps[order][ok].max()] = np.nan
            gd2[grid > steps[order][ok].max()] = np.nan
        per_n[d["n_d1"]]["cur"].append(gcur)
        per_n[d["n_d1"]]["d2"].append(gd2)
    for n in uniq_ns:
        k = len(per_n[n]["cur"])
        axes[0].plot(grid / 1e6, np.nanmean(per_n[n]["cur"], axis=0), color=color_of[n], lw=2.2, label=f"|O|={n} (mean of {k})")
        axes[1].plot(grid / 1e6, np.nanmean(per_n[n]["d2"], axis=0), color=color_of[n], lw=2.2, label=f"|O|={n} (mean of {k})")
    boundary = float(runs[0]["run"].config.get("total_timesteps_d1", 1e8)) / 1e6
    floor = max((d["floor"] for d in runs), default=-1.0)
    for ax, title in zip(
        axes,
        ["current-round bank success (d1 in round 1, d2 in round 2)", "held-out d2 bank success (zero-shot during round 1)"],
    ):
        ax.axvline(boundary, ls="--", color="gray", lw=1)
        if floor >= 0:
            ax.axhline(floor, ls=":", color="red", lw=1.2, label=f"null-policy floor ({floor:.2f})")
        ax.set_xlabel("env steps (M)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"success rate (depths {','.join(map(str, depths))})")
    axes[0].legend()
    fig.suptitle(f"Kinetix-Banyan [{tag}]: success vs steps (d1→d2 boundary at {boundary:.0f}M)\n{PRELIM}", fontsize=9)
    fig.tight_layout()
    p1 = args.outdir / "kinetix_figure5_success_vs_steps.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---------- Panel 2: delta_2 vs |O| ----------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    u, m, lo, hi = _agg(ns, delta2)
    ax.errorbar(u, m, yerr=[m - lo, hi - m], fmt="o-", capsize=3, label=f"Δ₂ (mean, min–max over {len(runs)} runs)")
    ax.scatter(ns, delta2, s=14, color="k", alpha=0.5, zorder=3, label="individual seeds")
    if len(depths) > 1:
        ax.plot(ns, delta2_d1, "s--", alpha=0.6, label="Δ₂ depth-1")
        ax.plot(ns, delta2_d2, "^--", alpha=0.6, label="Δ₂ depth-2")
    ax.set_xscale("log")
    ax.set_xlabel("|O| (d1 object assignments)")
    ax.set_ylabel("Δ₂ = S_end(d1) − S_start(d2)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Forward-transfer gap vs diversity [{tag}]\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p2 = args.outdir / "kinetix_figure5_delta2_vs_O.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---------- Panel 3: B(2,1) vs |O| ----------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    u, m, lo, hi = _agg(ns, b21)
    ax.errorbar(u, m, yerr=[m - lo, hi - m], fmt="o-", capsize=3, label=f"B(2,1) (mean, min–max over {len(runs)} runs)")
    ax.scatter(ns, b21, s=14, color="k", alpha=0.5, zorder=3, label="individual seeds")
    if len(depths) > 1:
        ax.plot(ns, b21_d1, "s--", alpha=0.6, label="depth-1")
        ax.plot(ns, b21_d2, "^--", alpha=0.6, label="depth-2")
    ax.set_xscale("log")
    ax.set_xlabel("|O| (d1 object assignments)")
    ax.set_ylabel("B(2,1) = S_end_final(d1) − S_end(d1)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Backward transfer vs diversity [{tag}]\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p3 = args.outdir / "kinetix_figure5_B21_vs_O.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ---------- Panel 4: dead-end rate + type selectivity ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, d in enumerate(runs):
        color = color_of[d["n_d1"]]
        dg = d["diag"]
        if dg:
            steps = np.array([row["timing/num_env_steps"] for row in dg], dtype=float)
            de = np.array(
                [0.5 * (row["eval/d1_deadend_depth2"] + row["eval/d2_deadend_depth2"]) for row in dg],
                dtype=float,
            )
            axes[0].plot(steps / 1e6, de, color=color, label=f"|O|={d['n_d1']}")
        sl = d["sel"]
        if sl:
            steps = np.array([row["timing/num_env_steps"] for row in sl], dtype=float)
            ratio = np.array(
                [
                    row["train/n_required_in_zone"] / max(row["train/n_in_zone"], 1e-6)
                    for row in sl
                ],
                dtype=float,
            )
            axes[1].plot(steps / 1e6, np.clip(ratio, 0, 1), color=color, label=f"|O|={d['n_d1']}")
    axes[0].set_title("depth-2 dead-end rate (banks avg)")
    axes[1].set_title("type selectivity: required / delivered (train)")
    axes[1].axhline(0.5, ls=":", color="gray", lw=1, label="chance (~0.5)")
    for ax in axes:
        ax.axvline(boundary, ls="--", color="gray", lw=1)
        ax.set_xlabel("env steps (M)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Diagnostics: dead-ends and selectivity\n{PRELIM}", fontsize=9)
    fig.tight_layout()
    p4 = args.outdir / "kinetix_figure5_diagnostics.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    # ---------- CSV ----------
    csv_path = args.outdir / "kinetix_figure5_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["n_d1", "seed", "S_end_d1", "S_start_d2", "S_end_d2", "S_end_final_d1", "delta_2", "delta_2_depth1", "delta_2_depth2", "B_2_1", "B_2_1_depth1", "B_2_1_depth2", "overlap", "run_id"]
        )
        for i, d in enumerate(runs):
            w.writerow([ns[i], d["seed"], _summ(d, "boundary/S_end_d1"), _summ(d, "boundary/S_start_d2"), _summ(d, "final/S_end_d2"), _summ(d, "final/S_end_final_d1"),
                        delta2[i], delta2_d1[i], delta2_d2[i], b21[i], b21_d1[i], b21_d2[i], overlaps[i], d["run"].id])

    print("delta_2 by |O|:", dict(zip(ns, np.round(delta2, 3))))
    print("B_2_1  by |O|:", dict(zip(ns, np.round(b21, 3))))
    print(f"wrote {p1}, {p2}, {p3}, {csv_path}")
    if args.no_wandb:
        return

    # ---------- Log to W&B ----------
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        name=f"figure5_plots_{tag}",
        notes=f"Figure-5 panels for the Kinetix substrate from group {args.group}. {PRELIM}",
        job_type="analysis",
    )
    run.log(
        {
            "figures/kinetix_figure5_success_vs_steps": wandb.Image(str(p1)),
            "figures/kinetix_figure5_delta2_vs_O": wandb.Image(str(p2)),
            "figures/kinetix_figure5_B21_vs_O": wandb.Image(str(p3)),
            "figures/kinetix_figure5_diagnostics": wandb.Image(str(p4)),
        }
    )
    table = wandb.Table(
        columns=["n_d1", "seed", "delta_2", "delta_2_depth1", "delta_2_depth2", "B_2_1", "B_2_1_depth1", "B_2_1_depth2", "run_id"],
        data=[
            [ns[i], runs[i]["seed"], delta2[i], delta2_d1[i], delta2_d2[i], b21[i], b21_d1[i], b21_d2[i], runs[i]["run"].id]
            for i in range(len(runs))
        ],
    )
    run.log({"figures/metrics_table": table})
    # native scatter panels
    for i in range(len(runs)):
        run.log(
            {
                "fig5/n_d1": ns[i],
                "fig5/delta_2": delta2[i],
                "fig5/delta_2_depth2": delta2_d2[i],
                "fig5/B_2_1": b21[i],
            }
        )
    art = wandb.Artifact("kinetix_figure5_data", type="analysis")
    art.add_file(str(csv_path))
    for p in (p1, p2, p3, p4):
        art.add_file(str(p))
    run.log_artifact(art)
    run.finish()
    print("delta_2 by |O|:", dict(zip(ns, np.round(delta2, 3))))
    print("B_2_1  by |O|:", dict(zip(ns, np.round(b21, 3))))
    print(f"wrote {p1}, {p2}, {p3}, {csv_path}; logged to wandb run figure5_plots")


if __name__ == "__main__":
    main()
