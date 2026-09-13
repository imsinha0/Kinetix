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
* 2026-09-13 14:05: kempner_h100 queue blocked by 72 pending fig6_netsize jobs (user's own Banyan sweep, submitted 10:32) + 16 running = per-user cap. Resubmitted the three runs to `kempner_requeue` (preemptible, PreemptMode=REQUEUE, `--requeue`); n=1 and n=256 started within 2 min (holygpu7c nodes). If preemptions bite, add round-boundary checkpoint/resume.
* 14:15: requeue partition preempted both running jobs within minutes (holygpu7c nodes) — unusable for 8h runs even with resume (each restart re-compiles ~3–5 min). Added chunk-level checkpoint/resume to `ppo_rounds.py` (params, optimizer, RMS, boundary matrix, counters, W&B run id → `checkpoints/rounds/<group>/<run>_s<seed>/latest.pkl`; a requeued job continues mid-round in the same W&B run; tested on CPU). Resubmitted all three to kempner_h100 and used `scontrol top` to place them ahead of the user's 72 pending fig6_netsize jobs (same user, so this only reorders our own queue).
