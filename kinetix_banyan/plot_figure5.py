"""Phase 5: recreate Figure 5 for the Kinetix substrate from the sweep runs.

Pulls the 4 fig5 runs from W&B (group kinetix-banyan-fig5), extracts eval
histories + transfer metrics, renders the three panels, and logs everything
back to W&B (figures as images, metrics as native panels, data as CSV).

  python -m kinetix_banyan.plot_figure5 [--group kinetix-banyan-fig5]
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
PRELIM = "PRELIMINARY — 1 seed per point, Kinetix substrate. Pattern only. Not for publication."


def fetch_runs(group: str):
    api = wandb.Api()
    runs = [
        r
        for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
        if r.state == "finished"
    ]
    out = []
    for r in runs:
        cfg = {k: v for k, v in r.config.items()}
        n_d1 = int(cfg.get("d1_num_instances", -1))
        summ = dict(r.summary)
        hist = r.history(
            keys=[
                "timing/num_env_steps",
                "round",
                "eval/d1_success_depth1",
                "eval/d1_success_depth2",
                "eval/d2_success_depth1",
                "eval/d2_success_depth2",
            ],
            pandas=False,
        )
        diag = r.history(
            keys=[
                "timing/num_env_steps",
                "eval/d1_deadend_depth2",
                "eval/d2_deadend_depth2",
            ],
            pandas=False,
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
            )
        )
    out.sort(key=lambda d: d["n_d1"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="kinetix-banyan-fig5")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/fig5"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = fetch_runs(args.group)
    assert runs, f"no finished runs in group {args.group}"
    print(f"found {len(runs)} runs: |O| = {[d['n_d1'] for d in runs]}")

    ns = [d["n_d1"] for d in runs]
    delta2 = [float(d["summary"].get("transfer/delta_2", np.nan)) for d in runs]
    delta2_d1 = [float(d["summary"].get("transfer/delta_2_depth1", np.nan)) for d in runs]
    delta2_d2 = [float(d["summary"].get("transfer/delta_2_depth2", np.nan)) for d in runs]
    b21 = [float(d["summary"].get("transfer/B_2_1", np.nan)) for d in runs]
    b21_d1 = [float(d["summary"].get("transfer/B_2_1_depth1", np.nan)) for d in runs]
    b21_d2 = [float(d["summary"].get("transfer/B_2_1_depth2", np.nan)) for d in runs]
    overlaps = [int(d["summary"].get("diversity/d1_d2_overlap", d["run"].config.get("diversity/d1_d2_overlap", -1))) for d in runs]
    assert all(o == 0 for o in overlaps), f"d1/d2 overlap nonzero: {overlaps}"

    # ---------- Panel 1: success vs env steps, one line per |O| ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(runs):
        color = cmap(i / max(1, len(runs) - 1))
        h = d["history"]
        steps = np.array([row["timing/num_env_steps"] for row in h], dtype=float)
        rounds = np.array([row.get("round", 1) for row in h])
        # During round 1 the "current bank" is d1; during round 2 it's d2.
        cur = np.array(
            [
                0.5
                * (
                    (row["eval/d1_success_depth1"] if row.get("round", 1) == 1 else row["eval/d2_success_depth1"])
                    + (row["eval/d1_success_depth2"] if row.get("round", 1) == 1 else row["eval/d2_success_depth2"])
                )
                for row in h
            ],
            dtype=float,
        )
        d2s = np.array(
            [
                0.5 * (row["eval/d2_success_depth1"] + row["eval/d2_success_depth2"])
                for row in h
            ],
            dtype=float,
        )
        axes[0].plot(steps / 1e6, cur, color=color, label=f"|O|={d['n_d1']}")
        axes[1].plot(steps / 1e6, d2s, color=color, label=f"|O|={d['n_d1']}")
    boundary = float(runs[0]["run"].config.get("total_timesteps_d1", 1e8)) / 1e6
    floor = max((d["floor"] for d in runs), default=-1.0)
    for ax, title in zip(axes, ["current-round bank success", "held-out d2 bank success"]):
        ax.axvline(boundary, ls="--", color="gray", lw=1)
        if floor >= 0:
            ax.axhline(floor, ls=":", color="red", lw=1.2, label=f"null-policy floor ({floor:.2f})")
        ax.set_xlabel("env steps (M)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (depths 1-2 avg)")
    axes[0].legend()
    fig.suptitle(f"Kinetix-Banyan: success vs steps (d1→d2 boundary at {boundary:.0f}M)\n{PRELIM}", fontsize=9)
    fig.tight_layout()
    p1 = args.outdir / "kinetix_figure5_success_vs_steps.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---------- Panel 2: delta_2 vs |O| ----------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(ns, delta2, "o-", label="Δ₂ (avg)")
    ax.plot(ns, delta2_d1, "s--", alpha=0.6, label="Δ₂ depth-1")
    ax.plot(ns, delta2_d2, "^--", alpha=0.6, label="Δ₂ depth-2")
    ax.set_xscale("log")
    ax.set_xlabel("|O| (d1 object assignments)")
    ax.set_ylabel("Δ₂ = S_end(d1) − S_start(d2)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Forward-transfer gap vs diversity\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p2 = args.outdir / "kinetix_figure5_delta2_vs_O.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---------- Panel 3: B(2,1) vs |O| ----------
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(ns, b21, "o-", label="B(2,1) (avg)")
    ax.plot(ns, b21_d1, "s--", alpha=0.6, label="depth-1")
    ax.plot(ns, b21_d2, "^--", alpha=0.6, label="depth-2")
    ax.set_xscale("log")
    ax.set_xlabel("|O| (d1 object assignments)")
    ax.set_ylabel("B(2,1) = S_end_final(d1) − S_end(d1)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Backward transfer vs diversity\n{PRELIM}", fontsize=8)
    fig.tight_layout()
    p3 = args.outdir / "kinetix_figure5_B21_vs_O.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ---------- Panel 4: dead-end rate + type selectivity ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, d in enumerate(runs):
        color = cmap(i / max(1, len(runs) - 1))
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
            ["n_d1", "delta_2", "delta_2_depth1", "delta_2_depth2", "B_2_1", "B_2_1_depth1", "B_2_1_depth2", "overlap", "run_id"]
        )
        for i, d in enumerate(runs):
            w.writerow([ns[i], delta2[i], delta2_d1[i], delta2_d2[i], b21[i], b21_d1[i], b21_d2[i], overlaps[i], d["run"].id])

    # ---------- Log to W&B ----------
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        name="figure5_plots",
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
        columns=["n_d1", "delta_2", "delta_2_depth1", "delta_2_depth2", "B_2_1", "B_2_1_depth1", "B_2_1_depth2", "run_id"],
        data=[
            [ns[i], delta2[i], delta2_d1[i], delta2_d2[i], b21[i], b21_d1[i], b21_d2[i], runs[i]["run"].id]
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
