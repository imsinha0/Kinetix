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
* 19:05: cancelled d1v1 o10/o100/o1000_nd0 (lip 0.55, 0% at 35M; the nd0 arm cannot show a diversity effect anyway) to free the per-user GPU cap (QOSMaxGRESPerUser) for d1-v2. Kept o1_nd0, o1_nd0_lip0 and all nd3 runs.

#### d1-v1 boundary + d1-v2 at ~60M (2026-09-05 19:45)
* d1-v1 lip 0.55: every run (nd0 and nd3, all |O|) reached the boundary with S_end(d1)=0, S_start(d2)=0; the o10_nd3 flinger decayed back to 0. Cancelled all except o1_nd0_lip0 (S_end_d1=1.0, S_start_d2=0.994 → Δ₂≈0.006 at |O|=1: no gap when identity is irrelevant).
* d1-v2 (lip 0, **terminal** −1 wrong deposit): untrained floor = bulldoze chance (nd1 ≈0.56, nd3 ≈0.27). Then every run except one collapsed to 0%: the agent learns to avoid the zone (0 beats expected −0.5). **o1_nd1_lip0_pen (l1vy1de4)**: 100% on d1 by 17M, **81% on novel-object d2** (flat) → identity is used and a forward gap Δ₂≈0.19 exists at |O|=1. Cancelled the 0% runs; kept l1vy1de4 and the two pending lip-0.25 runs.
* Fix: make the penalty **non-terminal** (as in Point Mass, REWARD_WRONG_DEPOSIT=−0.02 there), charged once per distractor entering the zone, p=0.3: bulldozing k distractors pays 1−k·p (>0, so approach is never extinguished) while selecting pays 1.

### Batch d1-v3 (submitted 2026-09-05 19:55, SLURM 44684011-44684033) — depth 1, lip 0, non-terminal `reward_wrong_deposit=0.3`, 100M+100M
| job | name | |O| | distr. |
|---|---|---|---|
| 44684011 | o1_nd1_p03 | 1 | 1 |
| 44684024 | o10_nd1_p03 | 10 | 1 |
| 44684027 | o100_nd1_p03 | 100 | 1 |
| 44684028 | o1000_nd1_p03 | 1000 | 1 |
| 44684029 | o1_nd3_p03 | 1 | 3 |
| 44684030 | o10_nd3_p03 | 10 | 3 |
| 44684031 | o100_nd3_p03 | 100 | 3 |
| 44684033 | o1000_nd3_p03 | 1000 | 3 |

#### d1-v3 at ~75M / d1-v2 survivor at boundary (2026-09-05 20:55)
* **v2 o1_nd1 terminal −1 (l1vy1de4)** reached the boundary: S_end(d1)=1.000, S_start(d2)=0.812 → **Δ₂=0.19 at |O|=1** (first real forward gap). Video: arm reaches down and sweeps the right object onto the platform. Round 2 running.
* **v3 non-terminal p=0.3**: untrained greedy floor is already 0.7–0.98 (success no longer needs selectivity), so success rate cannot show identity use. Learned runs: o1000_nd1 100/100 (train return ≈1.03, n_in_zone≈1.09 → it *is* selective: distractor enters only ~9% of episodes, and this generalises to novel objects), o100_nd1 100/100, o10_nd1 100/99, o100_nd3 77/73 (flat, bulldozing), o1_nd3 ~0.75. **3 of 8 collapsed to 0% by 5–30M** (o1_nd1, o10_nd3, o1000_nd3) — agent stops interacting; seed-level instability, not |O|-related. Cancelled those and the two lip-0.25 runs (≤8%, no learning).
* Reading: a *terminal* penalty makes success identity-sensitive (needed for the figure); −1 was too harsh (avoidance). Hybrid = terminal but small.

### Batch d1-v4 (submitted 2026-09-05 20:55, SLURM 44694528-44694537) — depth 1, lip 0, 1 distractor, **terminal** `reward_wrong_deposit=0.3`, 100M+100M, seeds 0 & 1
Expected: bulldoze pays 0.5·1 − 0.5·0.3 = +0.35 (approach not extinguished); selecting pays +1. Success = goal object enters before the distractor.
|O| ∈ {1, 10, 100, 1000} × seed ∈ {0, 1}: o{O}_nd1_t03_s{seed}. Two jobs queued behind the per-user GPU cap.
Grasp arm (lip 0.55 + `reward_goal_distance_scale=0.2`) implemented and tested, to launch when GPUs free.
* **v2 o1_nd1 terminal −1 (l1vy1de4) FINISHED**: S_end(d1)=1.000, S_start(d2)=0.812, S_end(d2)=0.812, S_end_final(d1)=1.000 → **Δ₂=0.188, B(2,1)=0.000**. Notably d2 success did not move at all during 100M steps of round-2 training (train success on d2 ≈0.80): the |O|=1 policy is frozen — no plasticity for the novel objects. First complete data point of the figure (|O|=1).

#### d1-v4 at ~82M, pre-boundary (2026-09-05 21:50) — the Figure-5 pattern appears
| |O| | seed | S(d1) | S(d2) | gap |
|---|---|---|---|---|
| 1 | 0 | 1.000 | 0.812 | 0.19 (replicates v2) |
| 1 | 1 | collapsed (0/0 from start) | | |
| 10 | 0 | 1.000 | 0.773 | 0.23 |
| 10 | 1 | 1.000 | 0.531 | 0.47 |
| 100 | 0 | 0.859 | 0.797 | 0.06 |
| 100 | 1 (41M) | 0.906 | 0.812 | 0.09 |
| 1000 | 0 | 0.883 | 0.812 | 0.07 |
| 1000 | 1 | queued | | |
Gap ≈0.2–0.5 at |O|≤10, ≈0.06–0.09 at |O|≥100; d1 mastery lower at high |O| (0.86–0.91 vs 1.0), as in the paper.
Caveat: d2 success saturates at 0.812 (104/128 fixed episodes) in several runs → part of the residual high-|O| gap may be a layout ceiling of the fixed d2 eval set, not transfer. v3 non-terminal o1000_nd1 / o100_nd1 (100% at boundary) collapsed to 0 during round 2 → cancelled; queued seed 2 for all four |O| of v4.

#### Root cause of the sudden collapses (2026-09-05 22:05) — FIXED
Every collapse (v1 o10_nd3, v2 ×5, v3 ×5, v4 o1_s1) is the same event: in one update `episode_return`, `loss/*`, `grad_norm` and `param_norm` all turn NaN simultaneously and never recover. The reward goes NaN first → the source is the environment: a physics blow-up (violent flinging) makes positions NaN; the shaping terms (height potential, distance) propagate NaN into the reward; one PPO step makes the parameters NaN. Stock Kinetix only terminated NaN episodes, it never sanitised the reward.
Fix (commit "NaN guard"): reward := 0 on physics-NaN steps, infos nan_to_num'd, `train/physics_nan_rate` logged, and `optax.zero_nans()` in the optimizer chain so a NaN gradient skips the update. Test added. Seed-2 v4 jobs cancelled and resubmitted on the fixed code (44707099-44707102); seeds 0/1 (old code, ~95M) left running — o1_s1 is already dead.

#### d1-v4 boundary (2026-09-05 22:20)
| |O| | seed | S_end(d1) | S_start(d2) | Δ₂ |
|---|---|---|---|---|
| 1 | 0 | 1.000 | 0.812 | 0.188 |
| 1 | 1 | NaN-dead (old code) | | – |
| 1 | 2 (30M, new code) | 1.000 | 0.320 | (0.68 so far) |
| 10 | 0 | 0.998 | 0.732 | 0.266 |
| 10 | 1 | 1.000 | 0.516 | 0.484 |
| 100 | 0 | 0.828 | 0.812 | 0.016 |
| 100 | 1 (89M) | 0.977 | 0.875 | (0.10 so far) |
| 1000 | 0 | 0.852 | 0.812 | 0.040 |
Δ₂ ≈ 0.19–0.48 at |O| ≤ 10, ≈ 0.02–0.10 at |O| ≥ 100 → **Figure-5 pattern on Kinetix**. Round 2 lifts d2 at |O|=10 to 0.91–0.94 (plasticity present), unlike the frozen |O|=1 v2 run — so 0.812 was not an eval ceiling.
v3 finished (non-terminal, identity-insensitive success): Δ₂ = 0.008–0.025 everywhere, as predicted; not usable for the figure.

### Batch d2-v1 (submitted 2026-09-05 22:40, SLURM 44713222-44713235) — **depth 2 only** (`task_depths=[2]`), lip 0, 1 distractor, terminal `reward_wrong_deposit=0.3`, 100M+100M, seed 0
Episode = 3 objects (the tree's two leaves + 1 distractor); success = both leaves in the zone; distractor in zone → terminal −0.3; a globally-valid-but-not-required pair → −1 (structural dead-end). d2 bank = novel objects AND novel rules. |O| ∈ {1, 10, 100, 1000}: d2_o{O}_nd1_t03_s0. Terminal penalty generalised to all depths (commit c0694bb, 21 tests pass). Cancelled the NaN-dead d1v4-o1-s1.
