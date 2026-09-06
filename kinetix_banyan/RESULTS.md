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

#### First Kinetix Figure 5 (interim, 5 finished d1-v4 runs, 2026-09-05 23:20) — `outputs/fig5/d1-v4/`
| |O| | seed | S_end(d1) | S_start(d2) | Δ₂ | B(2,1) |
|---|---|---|---|---|---|
| 1 | 0 | 1.000 | 0.812 | 0.188 | 0.000 |
| 10 | 0 | 0.998 | 0.732 | 0.266 | 0.000 |
| 10 | 1 | 1.000 | 0.516 | 0.484 | −0.006 |
| 100 | 0 | 0.828 | 0.812 | 0.016 | 0.000 |
| 1000 | 0 | 0.852 | 0.812 | 0.039 | −0.002 |
Still running: o1000_s1 (boundary 0.957/0.908 → Δ₂=0.049; then NaN-died in round 2, old code), o100_s1 (0.990/0.910 → Δ₂=0.080), o1_s2 (1.000/0.693 → Δ₂=0.307), o1000_s2, o100_s2 (~90M).
B(2,1) ≈ 0 everywhere at depth 1 (d1 stays mastered), consistent with the paper's small backward effect on the object axis.
* o10_s2 (new code) is a genuine non-learner, not NaN: episode length 256, nothing ever reaches the zone, return ≈0.09 (shaping only), entropy decays → passive policy after early dead-ends (deadend_rate 0.17 at 15M). Terminal −0.3 can still induce avoidance in some seeds (1 of 8 new-code runs so far).
* 23:25: cancelled o10_s2 (non-learner); queued seed 3 for all four |O| (d1v4-o{O}-s3) so every point has ≥3 live seeds after the two NaN deaths of the old-code runs.

#### 2026-09-06 00:25 — depth-1 sweep nearly complete; depth-2 boundary reached
**d1-v4 Δ₂ by |O| (S_end(d1) / S_start(d2))**:
| |O| | s0 | s1 | s2 | s3 (running) |
|---|---|---|---|---|
| 1 | 1.000/0.812 = **0.188** | NaN-dead | 1.000/0.693 = **0.307** | 1.000/0.812 @88M |
| 10 | 0.998/0.732 = **0.266** | 1.000/0.516 = **0.484** | non-learner | 0.758/0.805 @86M |
| 100 | 0.828/0.812 = **0.016** | 0.990/0.910 = **0.080** | 0.801/0.791 = **0.010** | 0.852/0.797 @88M |
| 1000 | 0.852/0.812 = **0.039** | 0.957/0.908 = **0.049** (NaN-died in r2) | 0.846/0.811 = **0.035** | 0.883/0.812 @69M |
B(2,1) ≈ 0 for all live runs (−0.09 for o1_s2). Two policy modes visible: "0.81-plateau" (never improves on d2 in round 2) and "0.98-mode" (o10_s0/s1, o100_s1 reach ≥0.98 on d2 in round 2).

**d2-v1 (depth 2 only, 3 objects) boundary, seed 0**:
| |O| | S_end(d1) | S_start(d2) | Δ₂ |
|---|---|---|---|
| 1 | 0.949 | 0.289 | **0.660** |
| 10 | 0.764 | 0.816 | −0.052 |
| 100 | 0.652 | 0.707 | −0.055 |
| 1000 | 0.727 | 0.744 | −0.017 |
Depth 2 gives the sharpest picture yet: a 0.66 gap at |O|=1 collapsing to ≈0 by |O|=10, with d1 mastery decreasing with diversity (0.95 → 0.65–0.76), exactly the paper's qualitative Figure 5. Queued seeds 1 and 2 for all four depth-2 points (d2v1-o{O}-s{1,2}).

#### 2026-09-06 01:35 — d2-v1 seed 0 FINISHED (depth 2, 3 objects) — `outputs/fig5/d2-v1/`
| |O| | S_end(d1) | S_start(d2) | S_end(d2) | S_end_final(d1) | Δ₂ | B(2,1) |
|---|---|---|---|---|---|---|
| 1 | 0.949 | 0.289 | 0.986 | 0.998 | **0.660** | 0.049 |
| 10 | 0.764 | 0.816 | 0.879 | 0.867 | −0.053 | 0.104 |
| 100 | 0.652 | 0.707 | 0.898 | 0.902 | −0.055 | 0.250 |
| 1000 | 0.727 | 0.744 | 0.949 | 0.951 | −0.018 | 0.225 |
Depth 2 reproduces the paper's Figure-5 shape cleanly: Δ₂ collapses from 0.66 to ≈0 once |O| ≥ 10. Backward transfer is positive and grows with |O| (0.05 → 0.25): round-2 training on novel objects/rules *improves* d1 at high diversity. Seed 1 at 99M agrees at |O|=1 (0.906/0.242 → 0.66).
d1-v4 seed 3 boundaries: o1 1.000/0.812 (0.19), o10 0.775/0.781 (−0.01, "0.78-mode" seed), o100 0.830/0.799 (0.03), o1000 0.850/0.803 (0.05).
* Depth-2 |O|=1 held-out curve: d2 success *falls* during round 1 (≈0.5 at 60M → 0.29 at the boundary) while d1 is being mastered — overfitting to the single task with its fixed object codes; the higher-|O| runs' d2 curves rise monotonically. Round 2 then recovers |O|=1 to 0.99. Depth-2 figures sent to the user (seed 0).

#### 2026-09-06 02:05 — d1-v4 COMPLETE (14 runs, 3–4 live seeds per |O|) — `outputs/fig5/d1-v4/`, W&B run figure5_plots_d1-v4
| |O| | Δ₂ per seed | mean Δ₂ | B(2,1) |
|---|---|---|---|
| 1 | 0.188, 0.307, 0.188 | **0.23** | 0.00, −0.09, 0.00 |
| 10 | 0.266, 0.484, −0.006 | **0.25** | 0.00, −0.01, 0.01 |
| 100 | 0.016, 0.080, 0.010, 0.031 | **0.034** | ≈0 |
| 1000 | 0.039, 0.049, 0.035, 0.047 | **0.042** | ≈0 |
Depth-1 Figure 5 on Kinetix: Δ₂ ≈ 0.23–0.25 at |O| ≤ 10 vs ≈ 0.04 at |O| ≥ 100 (6× smaller), B(2,1) ≈ 0 throughout. The one outlier (o10_s3, Δ₂≈0) is a weak learner that never left the 0.78 plateau on d1 either. Depth 2 (seed 0) shows the same shape with a larger low-|O| gap (0.66) and positive B(2,1) growing with |O|.

### Batch d12-v1 (submitted 2026-09-06 02:10) — **mixed depths [1,2]** (the paper's protocol: banks contain both depths, success averaged over depths), lip 0, 1 distractor, terminal 0.3, 100M+100M, seed 0, |O| ∈ {1,10,100,1000}: d12_o{O}_nd1_t03_s0.

#### 2026-09-06 03:10 — d2-v1 all 12 runs finished; d12-v1 (mixed depths) seed 0 finished
**d12-v1 (task_depths=[1,2], success averaged over depths), seed 0**:
| |O| | S_end(d1) | S_start(d2) | Δ₂ (avg) | Δ₂ depth1 | Δ₂ depth2 | B(2,1) | B depth2 |
|---|---|---|---|---|---|---|---|
| 1 | 0.856 | 0.362 | **0.494** | 0.475 | 0.514 | 0.046 | 0.092 |
| 10 | 0.611 | 0.617 | −0.006 | −0.012 | 0.000 | 0.199 | 0.383 |
| 100 | 0.673 | 0.675 | −0.002 | 0.021 | −0.025 | 0.118 | 0.236 |
| 1000 | 0.293 | 0.298 | −0.005 | −0.004 | −0.006 | 0.033 | 0.035 |
Same shape under the paper's mixed-depth protocol: Δ₂ ≈ 0.5 at |O|=1, ≈ 0 for |O| ≥ 10; B(2,1) positive at |O|=10–100 (driven by depth 2). |O|=1000 is a slow learner here (0.29 after 100M; 2000 bank rows) — the paper's "more diversity → lower mastery in a fixed budget" effect, exaggerated. Queued seeds 1 and 2 (d12v1-o{O}-s{1,2}).
