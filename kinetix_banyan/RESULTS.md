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
