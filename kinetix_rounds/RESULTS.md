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

#### 2026-09-13 15:10 — pilot round 1 result: default locomotion distribution is NOT learnable in 100M
* n=1 (single level, q3fe13y5): 0% success through 40M steps at 49k sps. GIF: the walker flips onto its back in the first steps and crawls inverted toward the goal (dense reward ≈0.25/episode), never touching it in 256 steps. n=256 (d5rp2duu): 2–3% on every pool = chance. Cancelled both.
* Motors are always on with joint limits (not the cause). Likely causes: the asymmetric extra leg destabilises the body; 25% lava floors (body touch = −1) make a quarter of levels much harder; 256 steps is short for 2.5 units of travel by a crawler.
### Pilots loco-pilot (1 round × 100M, seed 0), submitted 15:10
| job | variant |
|---|---|
| pilot_simple512_n256 | 2-leg symmetric walkers (no extra legs), no lava floor, horizon 512 |
| pilot_simple256_n256 | same, horizon 256 |
| pilot_nored512_n256 | default morphology (1 extra leg), no lava floor, horizon 512 |
| pilot_simple512_n1 | simple variant on a single fixed level |

#### 2026-09-13 17:40 — pilot results
| pilot | eval success @ end |
|---|---|
| simple512_n256 (2-leg, no lava, 512 steps, 256 levels) | 0.094 after 100M |
| simple256_n256 | 0.039 after 100M |
| nored512_n256 (default morphology, no lava) | 0.078 @ 85M |
| **simple512_n1 (one level)** | **1.000 @ 62M** (train 0.84) |
The simple 2-leg / no-lava / 512-step variant is learnable per level (the default extra-leg + 256-step one was not: 0% at 40M), but generalising across 256 morphologies is slow (9% in 100M). Decision: run the 10-round sweep on this variant anyway (S_end(d_r) vs r is exactly what Figure 6 measures, high-n rounds will show slow accumulation), and pilot Kinetix's stock random 's' distribution (with the stock no-op filter) as a faster-learning alternative family.
### Batch loco-r10-v2 (17:45): simple512 variant, n ∈ {1, 256, 65536}, seed 0, 10 × 100M. Pilot rand-pilot: random 's' levels, no-op filtered, n=256, 1 × 100M.

#### 2026-09-13 19:05 — round-1 progress, and the random pilot
* loco2 (simple512): n=1 train success 0.90 @71M but greedy eval 0.00 (pilot with the same level reached greedy 1.0 @62M → run-to-run divergence; watching). n=256 / n=65536: eval 5–10% @50–70M, slow as expected.
* **rand-pilot (stock random 's' levels, no-op filtered, n=256, 1 × 100M): eval 0.41 @8M → 0.61 @100M, train 0.59, 103k sps.** Learns fast and steadily → better substrate for the 10-round graph than locomotion.
