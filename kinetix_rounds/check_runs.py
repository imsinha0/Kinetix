"""Progress table for the rounds experiment.  python -m kinetix_rounds.check_runs --group loco-r10-v1 [--history]"""
import argparse

import wandb

ENTITY, PROJECT = "imsinha-harvard-university", "kinetix-rounds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="loco-r10-v1")
    ap.add_argument("--filter", default="")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    api = wandb.Api()
    runs = sorted(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.group}), key=lambda r: r.name)
    for r in runs:
        if args.filter not in r.name:
            continue
        s = r.summary
        R = int(r.config.get("num_rounds", 10))
        rd = int(s.get("round", 1) or 1)
        f = lambda k, p=3: (f"{float(s[k]):.{p}f}" if k in s and s[k] is not None else "-")
        steps = float(s.get("timing/num_env_steps", 0) or 0) / 1e6
        cur = f(f"eval/pool{rd}_success")
        print(
            f"{r.name.split('2048')[-1]:>16s} {r.id} {r.state:8s} {steps:7.1f}M round {rd}/{R} sps={float(s.get('timing/sps',0) or 0):6.0f} "
            f"train_succ={f('train/success_rate')} eval cur_pool={cur} pool1={f('eval/pool1_success')} "
            f"| S_end_mean={f('transfer/S_end_mean')} delta_mean={f('transfer/delta_mean')} B_R_1={f('transfer/B_R_1')}  {r.url}"
        )
        if args.history:
            keys = ["timing/num_env_steps", "round", "train/success_rate"] + [f"eval/pool{j}_success" for j in range(1, R + 1)]
            h = [x for x in r.history(keys=keys, pandas=False) if x.get("eval/pool1_success") is not None]
            step = max(1, len(h) // 12)
            for x in h[::step]:
                rr = int(x.get("round", 1) or 1)
                pools = " ".join(f"p{j}={x[f'eval/pool{j}_success']:.2f}" for j in range(1, R + 1) if f"eval/pool{j}_success" in x)
                print(f"      {x['timing/num_env_steps']/1e6:7.1f}M r{rr} train={x.get('train/success_rate', float('nan')):.2f} {pools}")


if __name__ == "__main__":
    main()
