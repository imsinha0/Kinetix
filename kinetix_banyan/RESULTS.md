# Kinetix-Banyan autoresearch log

Goal: reproduce Banyan paper Figure 5 (arXiv 2606.00880) on the Kinetix substrate —
forward-transfer gap Δ₂ = S_end(d1) − S_start(d2) and backward transfer
B(2,1) = S_end_final(d1) − S_end(d1), plotted against round-1 object-assignment
diversity |O|. Expected pattern: Δ₂ shrinks toward 0 as |O| grows.

Substrate: Kinetix `l/grasp_easy` claw arm; combine zone above the platform;
Banyan task banks from banyan-grid (`banyan_items_curriculum`, disjoint object
pools so every round-2 object type is novel). Type identity = fixed random
32-d code per object; goal = billboard circle carrying the goal token's code.
PPO (Kinetix stock, transformer entity model, 2048 envs, 64 steps), 2 rounds.
Trainer: `experiments/ppo_banyan.py`; task layer: `kinetix_banyan/task_layer.py`.

W&B: https://wandb.ai/imsinha-harvard-university/kinetix-banyan

## Prior state (Jul–Aug 2026, 4 objects = tree leaves + distractors)

| group | level | result |
|---|---|---|
| kinetix-banyan-fig5 (bhkudnpw, uwm85y7z, ko2li1p9, xantu30y) | no lip, 256 steps, sparse, 100M+100M | init policy already ~94% (constant torque bulldozes objects into zone). |O|=10 → 100% d1 depth-1 and 98% d2; |O|=1 collapsed to 0; |O|=100/1000 plateau ~48%. Δ₂≈0 everywhere → figure uninformative. |
| v2–v5 probes (ox3clneh … s9dzn4qs) | lip 0.55, 512 steps, shaping | 0% success everywhere after 50M. Lip made the task unlearnable in this budget. |

## Changes for the depth-1 program (2026-09-05)

* Episode = ONE task tree (as in Banyan). Depth 1 → exactly one object. Bank
  distractor slots inactive unless `num_distractors>0`.
* `task_depths: [1]` config restricts training/eval/metrics to depth 1.
* `object_condition: disjoint` (d2 object types are novel), horizon back to
  stock 256, eval every 16 updates. Shaping bonuses kept as they were
  (lift 0.1, deposit 0.2, height 0.4/m; identical at every |O|).
* Logged: `diversity/d1_distinct_goal_tokens`, `diversity/d1_d2_goal_token_overlap`.

## Runs

| id | group / name | |O| | distr. | lip | steps | S_end(d1) | S_start(d2) | Δ₂ | B(2,1) | notes / wandb |
|---|---|---|---|---|---|---|---|---|---|---|

### Incident 2026-09-05: first d1-v1 submission (44665145-44665173) ran on CPU
The venv's `jax_plugins/xla_cuda12/xla_cuda_plugin.so` (429 MB) was missing after the
move from lab storage; JAX silently fell back to CPU (~310 sps). Killed after 30 min,
W&B runs deleted, `jax[cuda12]==0.9.0` reinstalled, GPU verified via srun, and
`run_ppo.sh` now aborts if JAX has no GPU device. Resubmitted as 44670876-44670887.

### Batch d1-v1 (submitted 2026-09-05 18:20, SLURM 44670876-44670887) — depth 1, lip 0.55, 256 steps, 100M+100M, seed 0
Question: does depth-1 learn at all with the lip, and does |O| change Δ₂? Two arms: no distractors (nd0) vs 3 distractors (nd3, agent must read the goal billboard). Plus a lip-0 probe to measure how trivial the un-lipped task is.

| job | name | |O| | distr. | lip |
|---|---|---|---|---|
| 44670876 | o1_nd0 | 1 | 0 | 0.55 |
| 44670877 | o10_nd0 | 10 | 0 | 0.55 |
| 44670878 | o100_nd0 | 100 | 0 | 0.55 |
| 44670879 | o1000_nd0 | 1000 | 0 | 0.55 |
| 44670880 | o1_nd3 | 1 | 3 | 0.55 |
| 44670881 | o10_nd3 | 10 | 3 | 0.55 |
| 44670882 | o100_nd3 | 100 | 3 | 0.55 |
| 44670886 | o1000_nd3 | 1000 | 3 | 0.55 |
| 44670887 | o1_nd0_lip0 | 1 | 0 | 0 |

Early observation from the CPU run: at update 0 the lip-0 level already has train (stochastic) success 1.0 and eval (greedy) 0.117 — the un-lipped task is solved by random flailing, confirming the lip is needed for a learnable-but-nontrivial task.

Note: with disjoint pools and a 120-token vocab, each round has 60 object types, so distinct depth-1 goal types saturate at 60 for |O| ≥ 100 (logged as diversity/d1_distinct_goal_tokens).

#### d1-v1 at ~30M steps (2026-09-05 18:50)
* GPU throughput ~34k sps/run (≈1.6 h per 200M-step job).
* **lip 0 (o1_nd0_lip0, pa6kf57u)**: greedy eval 100% on d1 by 17M and 99% on the novel-object d2 bank *during round 1* → no code-overfitting at |O|=1 when the type is irrelevant. Confirms the no-distractor arm cannot produce the Figure-5 pattern: Δ₂≈0 trivially.
* **lip 0.55, nd0 (all |O|)**: 0% success, 0 objects ever in zone at 30M. Lift-from-floor not discovered (as in Aug probes).
* **lip 0.55, nd3**: 0% for |O|=1,100,1000; **o10_nd3 (zc1cvt63)** jumped to ~50% at 12.6M: ~1 object/episode in zone, goal 50% of the time (chance 25%) → partially selective. Lucky discovery, not reproducible across |O|.
* Conclusion: identity must matter for transfer to show (distractors), but the lip is the wrong anti-bulldozer (too hard). Replace it with Banyan's wrong-deposit penalty.

### Batch d1-v2 (submitted 2026-09-05 18:55, SLURM 44676068-44676086) — depth 1, `reward_wrong_deposit=1.0`, 100M+100M
Distractor in zone → terminal −1 (dead-end). Type-blind bulldozing now fails; the agent must read the goal billboard.

| job | name | |O| | distr. | lip |
|---|---|---|---|---|
| 44676068 | o1_nd3_lip0_pen | 1 | 3 | 0 |
| 44676071 | o10_nd3_lip0_pen | 10 | 3 | 0 |
| 44676075 | o100_nd3_lip0_pen | 100 | 3 | 0 |
| 44676076 | o1000_nd3_lip0_pen | 1000 | 3 | 0 |
| 44676078 | o1_nd1_lip0_pen | 1 | 1 | 0 |
| 44676080 | o1000_nd1_lip0_pen | 1000 | 1 | 0 |
| 44676084 | o1_nd3_lip025_pen | 1 | 3 | 0.25 |
| 44676086 | o1000_nd3_lip025_pen | 1000 | 3 | 0.25 |
