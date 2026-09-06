"""Quick W&B progress table for a group.  python -m kinetix_banyan.check_runs --group d1-v1 [--history]"""
import argparse

import wandb

ENTITY, PROJECT = "imsinha-harvard-university", "kinetix-banyan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="d1-v1")
    ap.add_argument("--filter", default="")
    ap.add_argument("--history", action="store_true", help="print the eval history per run")
    args = ap.parse_args()
    api = wandb.Api()
    runs = sorted(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.group}), key=lambda r: r.name)
    for r in runs:
        if args.filter not in r.name:
            continue
        s = r.summary
        dep = 2 if "eval/d1_success_depth2" in s and "eval/d1_success_depth1" not in s else 1
        f = lambda k, p=3: (f"{float(s[k]):.{p}f}" if k in s and s[k] is not None else "-")
        steps = float(s.get("timing/num_env_steps", 0) or 0) / 1e6
        print(
            f"{r.name.split('2048')[-1]:>16s} {r.id} {r.state:8s} {steps:6.1f}M r{s.get('round','-')} "
            f"sps={float(s.get('timing/sps',0) or 0):7.0f} train_succ={f('train/success_rate')} "
            f"ev{dep} d1={f(f'eval/d1_success_depth{dep}')} d2={f(f'eval/d2_success_depth{dep}')} "
            f"| S_end_d1={f('boundary/S_end_d1')} S_start_d2={f('boundary/S_start_d2')} "
            f"D2={f('transfer/delta_2')} B21={f('transfer/B_2_1')}  {r.url}"
        )
        if args.history:
            k1, k2 = f"eval/d1_success_depth{dep}", f"eval/d2_success_depth{dep}"
            h = r.history(keys=["timing/num_env_steps", "round", k1, k2], pandas=False)
            rows = [x for x in h if x.get(k1) is not None]
            step = max(1, len(rows) // 12)
            for x in rows[::step]:
                print(f"      {x['timing/num_env_steps']/1e6:6.1f}M r{x.get('round')} d1={x[k1]:.3f} d2={x[k2]:.3f}")


if __name__ == "__main__":
    main()
