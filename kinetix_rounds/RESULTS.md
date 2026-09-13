# Kinetix locomotion — 10-round diversity experiment

Goal: Banyan-paper Figure 6 on stock Kinetix tasks. 10 rounds × 100M steps; round r trains
on a disjoint pool of procedurally generated **locomotion** levels (`sample_locomotion_level`:
walker body + legs/ankles + extra legs with random dimensions, friction, density, motor
bindings; goal circle at random x; success = body reaches goal within 256 steps; stock dense
distance shaping 0.2). Pool diversity n ∈ {1, 256, 65536} distinct levels per round.
Level key = fold_in(fold_in(PRNGKey(task_seed), round), i) → reproducible, disjoint pools.

Metrics (from the boundary matrix M[b][j] = success on pool j after round b):
S_end(d_r) = M[r][r] (Fig-6 curve), Δ_r = M[r-1][r-1] − M[r-1][r], B(10,j) = M[10][j] − M[j][j].
All pools evaluated every 16 updates (64 episodes each) and in full (256) at boundaries.
GIFs: pool-1 and current-pool fixed episodes every 128 updates + boundaries (`video/round{r}/...`).

Code: `experiments/ppo_rounds.py`, `configs/ppo_rounds.yaml`, launch `kinetix_rounds/submit.sh`.
W&B project: https://wandb.ai/imsinha-harvard-university/kinetix-rounds

Level montage (pools 1–2, tasks 0–5): `outputs/rounds/locomotion_levels_pool1_pool2.png`.
Note ~25% of levels have a red (lava) floor: the green body must not touch it.

## Runs
### Batch loco-r10-v1 (2026-09-13) — n ∈ {1, 256, 65536}, seed 0, 10 × 100M
Round 1 doubles as the learnability pilot for the distribution.
