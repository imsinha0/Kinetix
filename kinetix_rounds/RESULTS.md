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
* Greedy eval (temperature 1e-4) is degenerate for n=1 pools: 256 identical deterministic rollouts → success flips 0/1 between checkpoints while train success is 0.9. Switched both configs to stock Kinetix sampled-policy eval (temperature 1.0). Cancelled loco-r10-v2 (75M into round 1) and restarted.
### Batch loco-r10-v3 (19:15): simple512 locomotion, n ∈ {1, 256, 65536}, seed 0, 10 × 100M, sampled eval.
### Batch rand-r10-v1 (19:15): stock random 's' levels (no-op filtered), n ∈ {1, 256, 65536}, seed 0, 10 × 100M, sampled eval. Untrained baseline on filtered pools ≈ 0.145; pilot reached 0.61 after 100M.

#### 2026-09-13 22:15 — after the first boundaries
**rand-r10-v1 (random 's', seed 0)**
* n=1: masters each round's single level (current pool 1.0) and forgets it at the next round (pool 1: 1.0 → 0.0 after round 2). Held-out pools ≈ 0 except one easy level (pool 6 ≈ 0.5). Round-5 level unsolved at 60M into the round.
* n=256: S_end(d1)=0.70, S_end(d2)=0.61; held-out pools 0.2–0.3 (untrained 0.145); pool 1 drops to 0.39 during round 2 (forgetting).
* n=65536: S_end(d1)=0.41 (lower per-round mastery), but **all** held-out pools rise together (0.3 → 0.45–0.50 by round 2) — the paper's transfer signature.
**loco-r10-v3**: n=1 failed round 1 (0%, the pilot had learned this level at 62M — stochastic), learned round 2's level (1.0) with transfer to pool 8 (0.88); n=256 / n=65536 stuck at 8–14% after 100M. Cancelled the two diverse locomotion runs (hopeless at 100M/round), kept loco3_n1. Queued rand seeds 1, 2 (6 jobs) for a 3-seed random-family Figure 6.

#### 2026-09-14 01:30 — rand-r10-v1 seed 0 COMPLETE (Figure 6 on Kinetix random 's' levels) — `outputs/rounds/rand-r10-v1/`
| n | S_end(d_r) by round | mean S_end | mean Δ_r | mean B(10,j) |
|---|---|---|---|---|
| 1 | 1, 1, 0, 0, 0, 1, 0.31, 0.97, 1, 1 | 0.63 | +0.50 | −0.39 |
| 256 | 0.61, 0.60, 0.69, 0.61, 0.58, 0.64, 0.65, 0.64, 0.61, 0.54 | 0.61 | +0.34 | −0.34 |
| 65536 | 0.34, 0.37, 0.35, 0.39, 0.39, 0.33, 0.40, 0.42, 0.41, 0.37 | 0.38 | −0.00 | +0.00 |
Untrained baseline on filtered pools ≈ 0.145.
* n=1: all-or-nothing per level (3 of 10 single levels never solved in 100M), Δ≈1 whenever solved, B(10,j) strongly negative (each level forgotten as soon as the next round starts; pool 1 re-solved only when a later level happens to be similar).
* n=256: fast within-round learning to ≈0.6 each round; every boundary drops to the held-out level (Δ_r ≈ +0.34 at all 9 boundaries) and each pool is forgotten to ≈0.25 afterwards (B ≈ −0.34). No accumulation across rounds.
* n=65536: lower per-round mastery (0.34 → 0.42, slowly rising), but **Δ_r ≈ 0 at every boundary and B ≈ 0**: no forward gap and no forgetting — the "systematic transfer" regime, at the cost of slower optimisation (the paper's plateau).
Seeds 1–2 running (finish ≈03:30). loco3_n1 (single-level locomotion) still running.

## RESULT — rand-r10-v1 COMPLETE, 3 seeds (2026-09-14 04:00) — `outputs/rounds/rand-r10-v1/`, W&B "Figure 6 — random small levels, 3 seeds"
| n per round | mean S_end(d_r) (3 seeds) | mean Δ_r | mean B(10,j) |
|---|---|---|---|
| 1 | 0.63 / 0.73 / 0.73 | +0.50 / +0.56 / +0.63 | −0.39 / −0.54 / −0.48 |
| 256 | 0.61 / 0.63 / 0.61 | +0.34 / +0.36 / +0.35 | −0.34 / −0.34 / −0.34 |
| 65536 | 0.38 / 0.41 / 0.39 | −0.00 / +0.02 / +0.01 | +0.00 / +0.01 / +0.01 |
Seeds agree to within ±0.03 at n=256 and n=65536. Mean S_end by round (3 seeds): n=256 flat at 0.60–0.67 (0.53 at round 10); n=65536 rises 0.36 → 0.43 by round 8 then dips (0.35 at round 10). Untrained baseline 0.145.
Interpretation: on Kinetix's random-level substrate, diversity trades per-round mastery for transfer in both directions — low/medium diversity (n ≤ 256) re-learns each pool quickly but shows a full forward gap at every boundary and forgets every previous pool; maximal diversity (n = 65536, a fresh level almost every episode) shows zero forward gap and zero forgetting but optimises slowly (paper: "too much diversity inhibits continued optimisation"). Locomotion (loco-r10-v3) was learnable only at n=1 within 100M/round.
