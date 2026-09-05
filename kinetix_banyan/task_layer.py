"""Banyan task layer on top of Kinetix (v3.0.0).

Reuses Kinetix's PPO, Jax2D physics, the ``l/grasp_easy`` arm level, and the
entity observation/rendering stack unchanged. Replaces ONLY the goal logic:

  * Each episode is ONE Banyan task tree (as in Banyan proper): exactly the
    tree's leaves are spawned as typed objects (circles) — one object at
    depth 1, two at depth 2 — with types drawn from a Banyan task bank (built
    by banyan-grid's ``build_d1_d2_task_banks``; same trees, same |O| sweep
    as Point Mass). ``num_distractors`` extra bank distractor objects (types
    never useful for the current tree) can be spawned alongside; default 0.
  * The fixated platform of grasp_easy (the stock "blue goal") becomes the
    COMBINE ZONE: a region above it, read directly by the reward function.
    Merges fire only when two objects are inside the zone simultaneously —
    never on incidental contact.
  * depth 1: the goal-typed object inside the zone -> +1, terminate.
    depth 2: the required (lhs, rhs) pair inside the zone -> +1, terminate;
    a globally-valid-but-not-required pair inside the zone -> -1, terminate
    (dead-end); anything else -> 0.
  * Reward is sparse: {0, +1, -1}. No shaping (dense_reward_scale must be 0).

Blocker-2 (role leakage) handling: every shape role is forced to 0, so the
role one-hot block in the observation is identically zero (asserted) and the
stock role-product reward can never fire. Type identity reaches the agent via
fixed random code vectors appended to the circle entity features; the goal
type is exposed as a fixated, non-colliding "billboard" circle carrying the
goal token's code. Codes are identity, constant across all tasks — they never
mark which pair is correct.

Tree information (Banyan's OBS_INCLUDE_TREE / child-tokens block): the
task's required rule (lhs, rhs -> out) is exposed as a second fixated
"rule billboard" circle whose extra feature columns carry the three codes
plus a valid flag. At depth 1 there is no rule and the block is all zeros.
"""

from __future__ import annotations

import functools
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import recalculate_mass_and_inertia
from jax2d.sim_state import CollisionManifold

from kinetix.environment.env import KinetixEnv
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.spaces import EntityObservations
from kinetix.util.saving import expand_env_state, load_from_json_file

# grasp_easy slot map (after expansion to NUM_CIRCLES below):
#   polygons: 0 floor, 1-2 side walls, 3 ceiling (fixated); 4-9 arm links /
#             claw fingers; 10 fixated platform (-> combine zone); 11 stock
#             carry box (deactivated).
#   circles:  0 arm anchor (fixated), 1 pedestal (fixated), 2 goal billboard,
#             3-6 typed object slots, 7 rule billboard (tree info).
BASE_LEVEL = "l/grasp_easy"
NUM_CIRCLES = 8
BILLBOARD_IDX = 2
OBJECT_SLOTS = (3, 4, 5, 6)
NUM_OBJECT_SLOTS = len(OBJECT_SLOTS)
STOCK_BOX_POLY_IDX = 11
ZONE_POLY_IDX = 10
OBJECT_RADIUS = 0.18
OBJECT_DENSITY = 1.0
ZONE_HEIGHT = 0.9
ZONE_MIN_HALF_WIDTH = 0.45
# Lip wall on the platform's left edge: objects cannot be PUSHED into the
# zone along the ground (a constant-torque "bulldozer" policy solved 78% of
# episodes type-blind without it); they must be lifted over. Reuses the
# deactivated stock-box polygon slot.
LIP_POLY_IDX = STOCK_BOX_POLY_IDX
LIP_HALF_WIDTH = 0.06
LIP_HEIGHT = 0.55
BILLBOARD_POSITION = (0.5, 4.5)
RULE_BILLBOARD_IDX = 7
RULE_BILLBOARD_POSITION = (1.5, 4.5)
TYPE_CODE_DIM = 32
# Rule block appended to every circle row: [code(lhs), code(rhs), code(out), valid]
RULE_BLOCK_DIM = 3 * TYPE_CODE_DIM + 1
TYPE_CODE_SEED = 1234


@struct.dataclass
class BanyanEnvState(EnvState):
    """EnvState + Banyan task fields (all per-episode, set by the reset fn).

    ``circle_types``: per-circle token id, -1 = untyped. Objects carry leaf
    tokens; the billboard carries the goal token. Polygons are never typed.
    """

    circle_types: jnp.ndarray = None  # (num_circles,) int32
    goal_token: jnp.ndarray = None  # scalar int32
    required_lhs: jnp.ndarray = None  # scalar int32, -1 at depth 1
    required_rhs: jnp.ndarray = None  # scalar int32, -1 at depth 1
    required_out: jnp.ndarray = None  # scalar int32, -1 at depth 1 (rule product = goal at depth 2)
    task_depth: jnp.ndarray = None  # scalar int32
    task_id: jnp.ndarray = None  # scalar int32
    # First-time event flags for exploration shaping (per object slot).
    lift_flags: jnp.ndarray = None  # (NUM_OBJECT_SLOTS,) bool
    deposit_flags: jnp.ndarray = None  # (NUM_OBJECT_SLOTS,) bool
    wrong_deposit_flags: jnp.ndarray = None  # (NUM_OBJECT_SLOTS,) bool: distractor already penalised
    # Highest y each object has reached (clipped at lip_top), for the dense
    # height-progress potential. Monotone -> bounded total shaping.
    max_heights: jnp.ndarray = None  # (NUM_OBJECT_SLOTS,) float32


@struct.dataclass
class BanyanTaskConstants:
    """Static (per-run) task data shared by reset, reward, and obs."""

    zone_lo: jnp.ndarray  # (2,) AABB lower corner
    zone_hi: jnp.ndarray  # (2,) AABB upper corner
    lip_top: jnp.ndarray  # scalar: world y of the lip's top edge
    spawn_positions: jnp.ndarray  # (NUM_OBJECT_SLOTS, 2)
    merge_lhs: jnp.ndarray  # (R,) canonical (min) side of global rulebook
    merge_rhs: jnp.ndarray  # (R,) canonical (max) side
    type_codes: jnp.ndarray  # (vocab + 1, TYPE_CODE_DIM); row 0 = untyped


def make_type_code_table(vocab_size: int, dim: int = TYPE_CODE_DIM) -> jnp.ndarray:
    """Fixed random ±1/sqrt(dim) identity codes; row 0 (untyped) is zeros."""
    rng = np.random.default_rng(TYPE_CODE_SEED)
    table = rng.choice([-1.0, 1.0], size=(vocab_size + 1, dim)).astype(np.float32)
    table /= np.sqrt(dim)
    table[0] = 0.0
    return jnp.asarray(table)


def _polygon_aabb(state: EnvState, idx: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(state.polygon.position[idx])
    nv = int(state.polygon.n_vertices[idx])
    verts = np.asarray(state.polygon.vertices[idx])[:nv]
    rot = float(state.polygon.rotation[idx])
    cos, sin = np.cos(rot), np.sin(rot)
    world = pos + verts @ np.array([[cos, sin], [-sin, cos]])
    return world.min(axis=0), world.max(axis=0)


def prepare_banyan_level(
    *,
    merge_lhs: np.ndarray,
    merge_rhs: np.ndarray,
    token_vocab_size: int,
    lip_height: float = LIP_HEIGHT,
) -> tuple[BanyanEnvState, StaticEnvParams, EnvParams, BanyanTaskConstants]:
    """Load grasp_easy, clear role semantics, add object/billboard slots, and
    derive the combine-zone geometry. Host-side, runs once."""
    state, static_env_params, env_params = load_from_json_file(BASE_LEVEL)
    static_env_params = static_env_params.replace(num_circles=NUM_CIRCLES)
    state = expand_env_state(state, static_env_params)

    # --- Blocker 2: erase every role. The stock reward needs role products
    # of 2 or 3; all-zero roles make it identically zero, and the role one-hot
    # block of the observation becomes constant zero.
    state = state.replace(
        polygon_shape_roles=jnp.zeros_like(state.polygon_shape_roles),
        circle_shape_roles=jnp.zeros_like(state.circle_shape_roles),
        polygon_highlighted=jnp.zeros_like(state.polygon_highlighted),
        circle_highlighted=jnp.zeros_like(state.circle_highlighted),
    )

    # Combine zone: the region above the fixated platform (stock blue goal).
    plat_lo, plat_hi = _polygon_aabb(state, ZONE_POLY_IDX)
    cx = 0.5 * (plat_lo[0] + plat_hi[0])
    half_w = max(0.5 * (plat_hi[0] - plat_lo[0]), ZONE_MIN_HALF_WIDTH)
    zone_lo = np.array([cx - half_w, plat_hi[1] - 0.05], dtype=np.float32)
    zone_hi = np.array([cx + half_w, plat_hi[1] + ZONE_HEIGHT], dtype=np.float32)

    # Repurpose the stock carry-box slot as the fixated lip wall guarding the
    # zone's left edge (see LIP_* constants above). lip_height <= 0 disables
    # the lip entirely (used by the lip-curriculum's first stage); the slot
    # then stays deactivated like the stock box we removed.
    poly = state.polygon
    if lip_height > 0.0:
        lip_cx = float(zone_lo[0]) - LIP_HALF_WIDTH
        lip_cy = float(plat_hi[1]) + 0.5 * lip_height
        lip_half = np.array([LIP_HALF_WIDTH, 0.5 * lip_height], dtype=np.float32)
        lip_verts = jnp.asarray(
            [
                [lip_half[0], lip_half[1]],
                [lip_half[0], -lip_half[1]],
                [-lip_half[0], -lip_half[1]],
                [-lip_half[0], lip_half[1]],
            ],
            dtype=jnp.float32,
        )
        poly = poly.replace(
            position=poly.position.at[LIP_POLY_IDX].set(jnp.asarray([lip_cx, lip_cy])),
            rotation=poly.rotation.at[LIP_POLY_IDX].set(0.0),
            velocity=poly.velocity.at[LIP_POLY_IDX].set(0.0),
            angular_velocity=poly.angular_velocity.at[LIP_POLY_IDX].set(0.0),
            vertices=poly.vertices.at[LIP_POLY_IDX].set(lip_verts),
            n_vertices=poly.n_vertices.at[LIP_POLY_IDX].set(4),
            inverse_mass=poly.inverse_mass.at[LIP_POLY_IDX].set(0.0),
            inverse_inertia=poly.inverse_inertia.at[LIP_POLY_IDX].set(0.0),
            friction=poly.friction.at[LIP_POLY_IDX].set(1.0),
            restitution=poly.restitution.at[LIP_POLY_IDX].set(0.0),
            collision_mode=poly.collision_mode.at[LIP_POLY_IDX].set(1),
            active=poly.active.at[LIP_POLY_IDX].set(True),
        )
    else:
        poly = poly.replace(active=poly.active.at[LIP_POLY_IDX].set(False))
    state = state.replace(polygon=poly)

    # Object spawn spots: the stock box position (in/near the claw, as the
    # level was designed) plus floor spots left of the platform. Objects
    # settle under gravity from these.
    floor_lo, floor_hi = _polygon_aabb(state, 0)
    y_floor = float(floor_hi[1]) + OBJECT_RADIUS + 0.05
    spawn_positions = np.array(
        [
            [0.74, 1.7],
            [1.5, y_floor],
            [2.15, y_floor],
            [2.8, y_floor],
        ],
        dtype=np.float32,
    )
    assert (spawn_positions[:, 0] < zone_lo[0]).all(), "spawn spot inside combine zone"

    # Billboards: fixated, non-colliding (collision_mode 0). The goal billboard
    # carries the goal token's type code; the rule billboard carries the
    # required rule's (lhs, rhs, out) codes (tree information). Both are always
    # present so the observation shape is constant across depths.
    circle = state.circle
    for idx, pos in ((BILLBOARD_IDX, BILLBOARD_POSITION), (RULE_BILLBOARD_IDX, RULE_BILLBOARD_POSITION)):
        circle = circle.replace(
            position=circle.position.at[idx].set(jnp.asarray(pos)),
            radius=circle.radius.at[idx].set(0.15),
            active=circle.active.at[idx].set(True),
            collision_mode=circle.collision_mode.at[idx].set(0),
            inverse_mass=circle.inverse_mass.at[idx].set(0.0),
            inverse_inertia=circle.inverse_inertia.at[idx].set(0.0),
        )

    # Typed object slots: dynamic circles (positions set per episode).
    obj = jnp.asarray(OBJECT_SLOTS)
    circle = circle.replace(
        radius=circle.radius.at[obj].set(OBJECT_RADIUS),
        active=circle.active.at[obj].set(True),
        collision_mode=circle.collision_mode.at[obj].set(1),
        friction=circle.friction.at[obj].set(1.0),
        restitution=circle.restitution.at[obj].set(0.0),
        # nonzero so recalculate_mass_and_inertia treats them as dynamic
        inverse_mass=circle.inverse_mass.at[obj].set(1.0),
        inverse_inertia=circle.inverse_inertia.at[obj].set(1.0),
    )
    state = state.replace(
        circle=circle,
        circle_densities=state.circle_densities.at[obj].set(OBJECT_DENSITY),
    )
    state = recalculate_mass_and_inertia(
        state, static_env_params, state.polygon_densities, state.circle_densities
    )

    env_params = env_params.replace(dense_reward_scale=0.0)

    constants = BanyanTaskConstants(
        zone_lo=jnp.asarray(zone_lo),
        zone_hi=jnp.asarray(zone_hi),
        lip_top=jnp.asarray(float(plat_hi[1]) + float(lip_height), dtype=jnp.float32),
        spawn_positions=jnp.asarray(spawn_positions),
        merge_lhs=jnp.minimum(jnp.asarray(merge_lhs), jnp.asarray(merge_rhs)).astype(jnp.int32),
        merge_rhs=jnp.maximum(jnp.asarray(merge_lhs), jnp.asarray(merge_rhs)).astype(jnp.int32),
        type_codes=make_type_code_table(token_vocab_size),
    )

    base_state = BanyanEnvState(
        **{f: getattr(state, f) for f in EnvState.__dataclass_fields__},
        circle_types=jnp.full((NUM_CIRCLES,), -1, dtype=jnp.int32),
        goal_token=jnp.asarray(-1, dtype=jnp.int32),
        required_lhs=jnp.asarray(-1, dtype=jnp.int32),
        required_rhs=jnp.asarray(-1, dtype=jnp.int32),
        required_out=jnp.asarray(-1, dtype=jnp.int32),
        task_depth=jnp.asarray(0, dtype=jnp.int32),
        task_id=jnp.asarray(-1, dtype=jnp.int32),
        lift_flags=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.bool_),
        deposit_flags=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.bool_),
        wrong_deposit_flags=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.bool_),
        max_heights=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.float32),
    )

    # Blocker-2 assertion: no shape carries a role.
    assert int(np.asarray(base_state.polygon_shape_roles).sum()) == 0
    assert int(np.asarray(base_state.circle_shape_roles).sum()) == 0
    return base_state, static_env_params, env_params, constants


def make_banyan_reset_fn(
    base_state: BanyanEnvState,
    constants: BanyanTaskConstants,
    task_bank: dict[str, jnp.ndarray],
    num_distractors: int = 0,
) -> Callable[[chex.PRNGKey], BanyanEnvState]:
    """Per-episode: sample a bank row, shuffle the tree's leaf objects over the
    spawn spots, and stamp types/goal/required-rule onto the base level.
    Fully vmappable. ``num_distractors`` of the bank's non-required (distractor)
    slots are activated per episode (chosen at random when fewer than
    available); 0 = the tree's leaves only.

    ``task_bank`` needs: leaf_token_ids (N, NUM_OBJECT_SLOTS), goal_token (N,),
    required_rule_lhs / required_rule_rhs (N, R>=1), required_rule_valid
    (N, R), depth (N,), task_id (N,).
    """
    n_tasks = int(task_bank["depth"].shape[0])
    leaf_tokens = jnp.asarray(task_bank["leaf_token_ids"], dtype=jnp.int32)
    required_mask = jnp.asarray(task_bank["required_leaf_mask"], dtype=jnp.bool_)
    assert leaf_tokens.shape[1] == NUM_OBJECT_SLOTS, (
        f"bank built with num_leaf_slots={leaf_tokens.shape[1]}, "
        f"scene has {NUM_OBJECT_SLOTS} object slots"
    )
    goal_tokens = jnp.asarray(task_bank["goal_token"], dtype=jnp.int32)
    rule_lhs = jnp.asarray(task_bank["required_rule_lhs"], dtype=jnp.int32)
    rule_rhs = jnp.asarray(task_bank["required_rule_rhs"], dtype=jnp.int32)
    rule_out = jnp.asarray(task_bank["required_rule_out"], dtype=jnp.int32)
    rule_valid = jnp.asarray(task_bank["required_rule_valid"], dtype=jnp.bool_)
    depths = jnp.asarray(task_bank["depth"], dtype=jnp.int32)
    task_ids = jnp.asarray(task_bank["task_id"], dtype=jnp.int32)
    obj = jnp.asarray(OBJECT_SLOTS)

    num_distractors = int(num_distractors)
    assert 0 <= num_distractors < NUM_OBJECT_SLOTS, num_distractors

    def reset(rng: chex.PRNGKey) -> BanyanEnvState:
        rng_task, rng_perm, rng_dis = jax.random.split(rng, 3)
        row = jax.random.randint(rng_task, (), 0, n_tasks)

        # Single tree per episode: the tree's own (required) leaves are always
        # spawned, plus up to ``num_distractors`` of the bank's distractor
        # slots (random subset). Everything else stays inactive and untyped.
        required = required_mask[row]
        score = jax.random.uniform(rng_dis, (NUM_OBJECT_SLOTS,)) + required  # required rank last
        rank = jnp.argsort(jnp.argsort(score))
        active = required | ((~required) & (rank < num_distractors))
        tokens = jnp.where(active, leaf_tokens[row], -1)
        perm = jax.random.permutation(rng_perm, NUM_OBJECT_SLOTS)
        positions = constants.spawn_positions[perm]

        circle = base_state.circle
        circle = circle.replace(
            position=circle.position.at[obj].set(positions),
            velocity=circle.velocity.at[obj].set(0.0),
            angular_velocity=circle.angular_velocity.at[obj].set(0.0),
            rotation=circle.rotation.at[obj].set(0.0),
            active=circle.active.at[obj].set(active),
        )

        circle_types = jnp.full((NUM_CIRCLES,), -1, dtype=jnp.int32)
        circle_types = circle_types.at[obj].set(jnp.where(active, tokens, -1))
        circle_types = circle_types.at[BILLBOARD_IDX].set(goal_tokens[row])

        # Depth 2 has exactly one required rule; depth 1 has none.
        has_rule = rule_valid[row, 0]
        lhs = jnp.where(has_rule, rule_lhs[row, 0], -1)
        rhs = jnp.where(has_rule, rule_rhs[row, 0], -1)
        out = jnp.where(has_rule, rule_out[row, 0], -1)

        return base_state.replace(
            circle=circle,
            circle_types=circle_types,
            goal_token=goal_tokens[row],
            required_lhs=jnp.minimum(lhs, rhs),
            required_rhs=jnp.maximum(lhs, rhs),
            required_out=out,
            task_depth=depths[row],
            task_id=task_ids[row],
            timestep=jnp.asarray(0, dtype=jnp.int32),
            last_distance=jnp.asarray(-1.0, dtype=jnp.float32),
            # An object spawning above the lip line (the claw-area spawn spot)
            # must not collect a free lift bonus while falling: mark its
            # first-lift as already consumed; same for the height potential
            # (its max starts at the spawn height).
            lift_flags=positions[:, 1] >= constants.lip_top,
            deposit_flags=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.bool_),
            wrong_deposit_flags=jnp.zeros((NUM_OBJECT_SLOTS,), dtype=jnp.bool_),
            max_heights=jnp.minimum(positions[:, 1], constants.lip_top).astype(jnp.float32),
        )

    return reset


class BanyanKinetixEnv(KinetixEnv):
    """KinetixEnv with the Banyan combine-zone goal instead of role products.

    Sparse reward in {0, +1, -1}; any nonzero reward terminates (engine_step's
    ``done = reward != 0``). Timeout handled by ``max_timesteps`` upstream.
    """

    def __init__(
        self,
        *args,
        constants: BanyanTaskConstants,
        reward_first_lift: float = 0.0,
        reward_first_deposit: float = 0.0,
        reward_height_scale: float = 0.0,
        reward_wrong_deposit: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_constants = constants
        # Banyan's wrong-deposit penalty (Point Mass: REWARD_WRONG_DEPOSIT, non-
        # terminal): -reward_wrong_deposit the first time each NON-required object
        # enters the combine zone; the episode continues. 0 = off. Makes identity
        # matter without a physical lip: flinging everything in still succeeds
        # but pays 1 - k*p, selecting the right object pays 1.
        self.reward_wrong_deposit = float(reward_wrong_deposit)
        # First-time exploration bonuses (one per required object per episode;
        # non-terminal; root success/dead-end stay +1/-1 terminal). 0 = pure
        # sparse. Same values at every |O| point keeps the Fig-5 sweep fair.
        self.reward_first_lift = float(reward_first_lift)
        self.reward_first_deposit = float(reward_first_deposit)
        # Dense height-progress potential: pays reward_height_scale per metre
        # of NEW maximum height (clipped at lip_top) for required objects.
        # Monotone in the per-object max -> total bounded by
        # scale * (lip_top - spawn_y) per required object (~0.3 at 0.4/m).
        self.reward_height_scale = float(reward_height_scale)

    def _banyan_reward_step(
        self,
        state: BanyanEnvState,
        manifolds: tuple[CollisionManifold, CollisionManifold, CollisionManifold],
    ):
        """Returns (state_with_updated_flags, reward, terminal, info)."""
        del manifolds  # merges fire on zone occupancy, never on contact
        c = self.task_constants
        obj = jnp.asarray(OBJECT_SLOTS)

        pos = state.circle.position[obj]
        in_zone = (
            state.circle.active[obj]
            & (pos[:, 0] >= c.zone_lo[0])
            & (pos[:, 0] <= c.zone_hi[0])
            & (pos[:, 1] >= c.zone_lo[1])
            & (pos[:, 1] <= c.zone_hi[1])
        )
        types = state.circle_types[obj]

        is_depth1 = state.required_lhs < 0

        # Depth 1: the goal-typed object inside the zone.
        d1_success = jnp.any(in_zone & (types == state.goal_token))

        # Depth 2: pairwise over object slots inside the zone.
        ii, jj = jnp.triu_indices(NUM_OBJECT_SLOTS, k=1)
        pair_in = in_zone[ii] & in_zone[jj]
        pair_lo = jnp.minimum(types[ii], types[jj])
        pair_hi = jnp.maximum(types[ii], types[jj])
        pair_required = pair_in & (pair_lo == state.required_lhs) & (pair_hi == state.required_rhs)
        pair_valid_global = pair_in & jnp.any(
            (pair_lo[:, None] == c.merge_lhs[None, :]) & (pair_hi[:, None] == c.merge_rhs[None, :]),
            axis=1,
        )
        d2_success = jnp.any(pair_required)
        d2_deadend = jnp.any(pair_valid_global & ~pair_required) & ~d2_success

        success = jnp.where(is_depth1, d1_success, d2_success)
        deadend = jnp.where(is_depth1, jnp.asarray(False), d2_deadend)
        terminal_reward = jnp.where(success, 1.0, jnp.where(deadend, -1.0, 0.0))
        terminal = success | deadend

        # First-time event bonuses for REQUIRED objects only (goal-typed at
        # depth 1; the rule pair at depth 2). Lifted = above the lip's top
        # (nothing in the scene rests that high except a carried object).
        is_required = jnp.where(
            is_depth1,
            types == state.goal_token,
            (types == state.required_lhs) | (types == state.required_rhs),
        ) & state.circle.active[obj]
        lifted_now = is_required & (pos[:, 1] >= c.lip_top)
        deposited_now = is_required & in_zone
        new_lifts = lifted_now & ~state.lift_flags
        new_deposits = deposited_now & ~state.deposit_flags
        heights = jnp.minimum(pos[:, 1], c.lip_top)
        height_gain = jnp.maximum(heights - state.max_heights, 0.0) * is_required
        # Wrong-deposit penalty: first entry of each non-required (distractor) object.
        wrong_now = in_zone & ~is_required
        new_wrong = wrong_now & ~state.wrong_deposit_flags
        bonus = (
            self.reward_first_lift * jnp.sum(new_lifts)
            + self.reward_first_deposit * jnp.sum(new_deposits)
            + self.reward_height_scale * jnp.sum(height_gain)
            - self.reward_wrong_deposit * jnp.sum(new_wrong)
        )
        state = state.replace(
            lift_flags=state.lift_flags | lifted_now,
            deposit_flags=state.deposit_flags | deposited_now,
            wrong_deposit_flags=state.wrong_deposit_flags | wrong_now,
            max_heights=jnp.maximum(state.max_heights, heights),
        )

        # Bonuses are dropped on terminal steps (success is exactly +1); the
        # wrong-deposit penalty is always charged so bulldozing never nets +1.
        reward = terminal_reward + jnp.where(terminal, 0.0, bonus) - jnp.where(
            terminal, self.reward_wrong_deposit * jnp.sum(new_wrong), 0.0
        )

        n_required_in_zone = jnp.where(
            is_depth1,
            jnp.sum(in_zone & (types == state.goal_token)),
            jnp.sum(in_zone & ((types == state.required_lhs) | (types == state.required_rhs))),
        )
        info = {
            "GoalR": success,
            "distance": 0.0,  # no dense shaping; dense_reward_scale is 0
            "deadend": deadend,
            "n_in_zone": jnp.sum(in_zone),
            "n_required_in_zone": n_required_in_zone,
            "n_wrong_in_zone": jnp.sum(wrong_now),
            "task_depth": state.task_depth,
        }
        return state, reward, terminal, info

    def compute_reward_info(
        self,
        state: BanyanEnvState,
        manifolds: tuple[CollisionManifold, CollisionManifold, CollisionManifold],
    ):
        """Kept for API compatibility (tests, parent-class callers); the real
        stepping path is engine_step below, which also carries flag updates."""
        _state, reward, _terminal, info = self._banyan_reward_step(state, manifolds)
        return reward, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def engine_step(self, env_state: BanyanEnvState, action_to_perform, env_params):
        """Parent engine_step with two Banyan changes: (1) done comes from the
        explicit terminal flag (success/dead-end), NOT from reward != 0, so
        non-terminal first-time bonuses don't end episodes; (2) the reward for
        a frame-skip block is the SUM of substep rewards up to and including
        the first terminal, so bonuses earned in the same block are kept."""

        def _single_step(env_state, unused):
            env_state, mfolds = self.physics_engine.step(
                env_state, env_params, action_to_perform
            )
            env_state, reward, terminal, info = self._banyan_reward_step(env_state, mfolds)
            return env_state, (reward, terminal, info)

        env_state, (rewards, terminals, infos) = jax.lax.scan(
            _single_step, env_state, xs=None, length=self.static_env_params.frame_skip
        )
        env_state = env_state.replace(timestep=env_state.timestep + 1)

        has_terminal = terminals.sum() > 0
        first_terminal_index = terminals.argmax()
        csum = jnp.cumsum(rewards)
        reward = jax.lax.select(has_terminal, csum[first_terminal_index], csum[-1])

        done = has_terminal | jax.tree.reduce(
            jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), env_state), False
        )
        done |= env_state.timestep >= env_params.max_timesteps

        info = jax.tree.map(
            lambda x: jax.lax.select(has_terminal, x[first_terminal_index], x[-1]), infos
        )

        return (
            jax.lax.stop_gradient(self.get_obs(env_state)),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            info,
        )

    def __hash__(self):
        # Reward params MUST be in the hash: jax.jit(static_argnums=(0,))
        # caches compiled step functions per env hash, and two envs differing
        # only in bonuses would otherwise silently share compiled rewards.
        return hash(
            (
                super().__hash__(),
                "banyan",
                self.reward_first_lift,
                self.reward_first_deposit,
                self.reward_height_scale,
                self.reward_wrong_deposit,
            )
        )

    def __eq__(self, value):
        return hash(self) == hash(value)


def make_banyan_env(
    *,
    base_state: "BanyanEnvState",
    static_env_params: StaticEnvParams,
    env_params: EnvParams,
    constants: BanyanTaskConstants,
    reset_fn: Callable[[chex.PRNGKey], "BanyanEnvState"],
    action_type: str = "multi_discrete",
    auto_reset: bool = True,
    reward_first_lift: float = 0.0,
    reward_first_deposit: float = 0.0,
    reward_height_scale: float = 0.0,
    reward_wrong_deposit: float = 0.0,
) -> "BanyanKinetixEnv":
    """Standard construction used by tests, the gate script, and PPO."""
    from kinetix.environment.spaces import (
        ContinuousActions,
        DiscreteActions,
        MultiDiscreteActions,
    )

    del base_state  # reserved for future use; reset_fn already closes over it
    action_cls = {
        "multi_discrete": MultiDiscreteActions,
        "continuous": ContinuousActions,
        "discrete": DiscreteActions,
    }[action_type]
    assert float(env_params.dense_reward_scale) == 0.0, "sparse reward only"
    return BanyanKinetixEnv(
        action_type=action_cls(env_params, static_env_params),
        observation_type=BanyanEntityObservations(env_params, static_env_params, constants),
        static_env_params=static_env_params,
        reset_function=reset_fn,
        auto_reset=auto_reset,
        constants=constants,
        reward_first_lift=reward_first_lift,
        reward_first_deposit=reward_first_deposit,
        reward_height_scale=reward_height_scale,
        reward_wrong_deposit=reward_wrong_deposit,
    )


class BanyanEntityObservations(EntityObservations):
    """Entity observations + per-circle type-code columns + tree/rule block.

    The base features already include the (all-zero) role one-hot; we append
    TYPE_CODE_DIM identity-code columns to the circle feature matrix. Objects
    carry their leaf token's code, the goal billboard carries the goal token's
    code, everything else (and inactive slots) is zero. A further RULE_BLOCK_DIM
    columns hold the required rule [code(lhs), code(rhs), code(out), valid] on
    the rule-billboard row only (zeros elsewhere, and all-zero at depth 1) —
    the analogue of Banyan's OBS_INCLUDE_TREE / child-tokens observation.
    """

    def __init__(self, env_params, static_env_params, constants: BanyanTaskConstants):
        super().__init__(env_params, static_env_params)
        self.task_constants = constants

    def get_obs(self, state: BanyanEnvState):
        base = super().get_obs(state)
        table = self.task_constants.type_codes
        codes = table[state.circle_types + 1] * state.circle.active[:, None]
        valid = (state.required_lhs >= 0).astype(jnp.float32)
        rule_row = jnp.concatenate(
            [table[state.required_lhs + 1], table[state.required_rhs + 1], table[state.required_out + 1], valid[None]]
        )
        rule = jnp.zeros((codes.shape[0], RULE_BLOCK_DIM), dtype=jnp.float32).at[RULE_BILLBOARD_IDX].set(rule_row)
        return base.replace(circles=jnp.concatenate([base.circles, codes, rule], axis=1))

    def observation_space(self, env_params):
        space = super().observation_space(env_params)
        circ = space.spaces["circles"]
        space.spaces["circles"] = type(circ)(
            -np.inf,
            np.inf,
            (circ.shape[0], circ.shape[1] + TYPE_CODE_DIM + RULE_BLOCK_DIM),
            circ.dtype,
        )
        return space
