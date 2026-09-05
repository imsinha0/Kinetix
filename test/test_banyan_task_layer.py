import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

BANYAN_GRID = Path(__file__).resolve().parents[2] / "banyan-grid"
if str(BANYAN_GRID) not in sys.path:
    sys.path.insert(0, str(BANYAN_GRID))

from benchmark.baselines.utils.continuous_banyan import build_d1_d2_task_banks

from kinetix_banyan.task_layer import (
    BILLBOARD_IDX,
    NUM_OBJECT_SLOTS,
    OBJECT_SLOTS,
    RULE_BILLBOARD_IDX,
    RULE_BLOCK_DIM,
    TYPE_CODE_DIM,
    BanyanEnvState,
    make_banyan_env,
    make_banyan_reset_fn,
    prepare_banyan_level,
)

ID_VOCAB = 24
MAX_DEPTH = 2


@pytest.fixture(scope="module")
def setup():
    d1_bank, d2_bank, meta = build_d1_d2_task_banks(
        d1_num_instances=4,
        d2_num_instances=4,
        num_leaf_slots=NUM_OBJECT_SLOTS,
        id_vocab_size=ID_VOCAB,
        recipe_length=MAX_DEPTH,
        seed=0,
        task_bank_mode="banyan_items_curriculum",
        object_condition="shared",
    )
    rulebook = meta["global_rulebook"]
    base_state, static_params, env_params, constants = prepare_banyan_level(
        merge_lhs=np.asarray(rulebook["merge_lhs"]),
        merge_rhs=np.asarray(rulebook["merge_rhs"]),
        token_vocab_size=int(meta["token_vocab_size"]),
    )
    reset_fn = make_banyan_reset_fn(base_state, constants, d1_bank)
    env = make_banyan_env(
        base_state=base_state,
        static_env_params=static_params,
        env_params=env_params,
        constants=constants,
        reset_fn=reset_fn,
        action_type="continuous",
    )
    return dict(
        env=env,
        env_params=env_params,
        static=static_params,
        base=base_state,
        constants=constants,
        d1_bank=d1_bank,
        d2_bank=d2_bank,
        meta=meta,
        reset_fn=reset_fn,
    )


def _zero_action(env):
    return jnp.zeros(env.action_space(None).shape, dtype=jnp.float32)


def _sc(x):
    return np.asarray(jax.device_get(x)).item()


def _zone_center(constants):
    lo = np.asarray(constants.zone_lo)
    hi = np.asarray(constants.zone_hi)
    return 0.5 * (lo + hi)


def _place_in_zone(state, slot_positions: dict[int, np.ndarray]):
    circle = state.circle
    for slot, pos in slot_positions.items():
        circle = circle.replace(
            position=circle.position.at[slot].set(jnp.asarray(pos, dtype=jnp.float32)),
            velocity=circle.velocity.at[slot].set(0.0),
        )
    return state.replace(circle=circle)


def test_roles_erased_and_billboard(setup):
    base = setup["base"]
    assert int(np.asarray(base.polygon_shape_roles).sum()) == 0
    assert int(np.asarray(base.circle_shape_roles).sum()) == 0
    # billboard fixated + non-colliding
    assert float(base.circle.inverse_mass[BILLBOARD_IDX]) == 0.0
    assert int(base.circle.collision_mode[BILLBOARD_IDX]) == 0
    # stock carry box slot repurposed as the fixated lip wall
    from kinetix_banyan.task_layer import LIP_POLY_IDX

    assert bool(base.polygon.active[LIP_POLY_IDX])
    assert float(base.polygon.inverse_mass[LIP_POLY_IDX]) == 0.0
    # lip guards the zone's left edge: its right face is at/left of zone_lo x
    lip_x = float(base.polygon.position[LIP_POLY_IDX][0])
    assert lip_x <= float(setup["constants"].zone_lo[0]) + 1e-4


def test_reset_stamps_task(setup):
    reset_fn = setup["reset_fn"]
    d1_bank = setup["d1_bank"]
    state = reset_fn(jax.random.PRNGKey(3))
    row = int(_sc(state.task_id))
    types = np.asarray(state.circle_types)
    active = np.asarray(state.circle.active)
    req = np.asarray(d1_bank["required_leaf_mask"])[row]
    bank_tokens = np.asarray(d1_bank["leaf_token_ids"])[row]
    slots = np.asarray(OBJECT_SLOTS)
    # Single tree per episode: exactly the tree's leaves are active and typed;
    # the bank's distractor slots are inactive and untyped.
    assert (active[slots] == req).all()
    assert (types[slots[req]] == bank_tokens[req]).all()
    assert (types[slots[~req]] == -1).all()
    assert types[BILLBOARD_IDX] == int(np.asarray(d1_bank["goal_token"])[row])
    depth = int(_sc(state.task_depth))
    assert int(active[slots].sum()) == depth  # 1 object at depth 1, 2 at depth 2
    if depth == 1:
        assert int(_sc(state.required_lhs)) == -1
    else:
        assert int(_sc(state.required_lhs)) >= 0
    # objects on distinct spawn spots
    pos = np.asarray(state.circle.position)[list(OBJECT_SLOTS)]
    assert len({tuple(np.round(p, 3)) for p in pos}) == NUM_OBJECT_SLOTS


def test_step_preserves_subclass_and_runs(setup):
    env, env_params, reset_fn = setup["env"], setup["env_params"], setup["reset_fn"]
    state = reset_fn(jax.random.PRNGKey(0))
    obs, next_state, reward, done, info = env.step(
        jax.random.PRNGKey(1), state, _zero_action(env), env_params
    )
    assert isinstance(next_state, BanyanEnvState)
    assert (np.asarray(next_state.circle_types) == np.asarray(state.circle_types)).all()
    assert float(reward) == 0.0 and not bool(done)
    assert "n_in_zone" in info and "deadend" in info


def _fresh_depth_state(setup, depth):
    bank = setup["d1_bank"]
    depths = np.asarray(bank["depth"])
    idx = int(np.nonzero(depths == depth)[0][0])
    row_bank = {k: v[idx : idx + 1] for k, v in bank.items()}
    reset_fn = make_banyan_reset_fn(setup["base"], setup["constants"], row_bank)
    return reset_fn(jax.random.PRNGKey(7))


def test_depth1_goal_in_zone_succeeds(setup):
    env, env_params = setup["env"], setup["env_params"]
    state = _fresh_depth_state(setup, 1)
    goal = int(_sc(state.goal_token))
    types = np.asarray(state.circle_types)
    goal_slot = [s for s in OBJECT_SLOTS if types[s] == goal][0]
    center = _zone_center(setup["constants"])
    state = _place_in_zone(state, {goal_slot: center})
    _obs, _st, reward, done, info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == 1.0 and bool(done)
    assert bool(np.asarray(info["GoalR"]))


def test_depth1_single_object_and_inactive_slot_no_reward(setup):
    env, env_params = setup["env"], setup["env_params"]
    state = _fresh_depth_state(setup, 1)
    types = np.asarray(state.circle_types)
    active = np.asarray(state.circle.active)
    assert sum(bool(active[s]) for s in OBJECT_SLOTS) == 1
    assert types[[s for s in OBJECT_SLOTS if active[s]][0]] == int(_sc(state.goal_token))
    # An inactive (distractor) slot dragged into the zone must not count.
    wrong_slot = [s for s in OBJECT_SLOTS if not active[s]][0]
    state = _place_in_zone(state, {wrong_slot: _zone_center(setup["constants"])})
    _obs, _st, reward, done, _info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == 0.0 and not bool(done)


def test_depth2_required_pair_merges(setup):
    env, env_params = setup["env"], setup["env_params"]
    state = _fresh_depth_state(setup, 2)
    lhs, rhs = int(_sc(state.required_lhs)), int(_sc(state.required_rhs))
    types = np.asarray(state.circle_types)
    slot_a = [s for s in OBJECT_SLOTS if types[s] == lhs][0]
    slot_b = [s for s in OBJECT_SLOTS if types[s] == rhs][0]
    center = _zone_center(setup["constants"])
    state = _place_in_zone(
        state, {slot_a: center + np.array([-0.15, 0.0]), slot_b: center + np.array([0.15, 0.0])}
    )
    _obs, _st, reward, done, info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == 1.0 and bool(done)
    assert int(np.asarray(info["n_required_in_zone"])) == 2


def test_depth2_single_required_object_nothing(setup):
    env, env_params = setup["env"], setup["env_params"]
    state = _fresh_depth_state(setup, 2)
    lhs = int(_sc(state.required_lhs))
    types = np.asarray(state.circle_types)
    slot_a = [s for s in OBJECT_SLOTS if types[s] == lhs][0]
    state = _place_in_zone(state, {slot_a: _zone_center(setup["constants"])})
    _obs, _st, reward, done, _info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == 0.0 and not bool(done)


def test_depth2_valid_but_not_required_pair_deadends(setup):
    env, env_params = setup["env"], setup["env_params"]
    constants = setup["constants"]
    state = _fresh_depth_state(setup, 2)
    req = (int(_sc(state.required_lhs)), int(_sc(state.required_rhs)))
    lhs_arr = np.asarray(constants.merge_lhs)
    rhs_arr = np.asarray(constants.merge_rhs)
    other = [
        (int(a), int(b)) for a, b in zip(lhs_arr, rhs_arr) if (int(a), int(b)) != req
    ][0]
    # stamp the two active object slots with the other rule's pair (synthetic scene)
    slots = [s for s in OBJECT_SLOTS if bool(np.asarray(state.circle.active)[s])]
    assert len(slots) == 2
    circle_types = state.circle_types.at[slots[0]].set(other[0]).at[slots[1]].set(other[1])
    state = state.replace(circle_types=circle_types)
    center = _zone_center(constants)
    state = _place_in_zone(
        state, {slots[0]: center + np.array([-0.15, 0.0]), slots[1]: center + np.array([0.15, 0.0])}
    )
    _obs, _st, reward, done, info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == -1.0 and bool(done)
    assert bool(np.asarray(info["deadend"]))
    assert not bool(np.asarray(info["GoalR"]))


def test_depth2_invalid_pair_nothing(setup):
    env, env_params = setup["env"], setup["env_params"]
    constants = setup["constants"]
    state = _fresh_depth_state(setup, 2)
    lhs_arr = np.asarray(constants.merge_lhs)
    rhs_arr = np.asarray(constants.merge_rhs)
    valid = {(int(a), int(b)) for a, b in zip(lhs_arr, rhs_arr)}
    pair = None
    for a in range(ID_VOCAB):
        for b in range(a + 1, ID_VOCAB):
            if (a, b) not in valid:
                pair = (a, b)
                break
        if pair:
            break
    slots = [s for s in OBJECT_SLOTS if bool(np.asarray(state.circle.active)[s])]
    assert len(slots) == 2
    circle_types = state.circle_types.at[slots[0]].set(pair[0]).at[slots[1]].set(pair[1])
    state = state.replace(circle_types=circle_types)
    center = _zone_center(constants)
    state = _place_in_zone(
        state, {slots[0]: center + np.array([-0.15, 0.0]), slots[1]: center + np.array([0.15, 0.0])}
    )
    _obs, _st, reward, done, _info = env.step(jax.random.PRNGKey(1), state, _zero_action(env), env_params)
    assert float(reward) == 0.0 and not bool(done)


def test_obs_type_codes_and_no_role_leak(setup):
    env, env_params, constants = setup["env"], setup["env_params"], setup["constants"]
    state = setup["reset_fn"](jax.random.PRNGKey(11))
    obs = env.get_obs(state)
    n_circ = setup["static"].num_circles
    assert obs.circles.shape == (n_circ, 19 + TYPE_CODE_DIM + RULE_BLOCK_DIM)
    codes = np.asarray(obs.circles[:, 19 : 19 + TYPE_CODE_DIM])
    table = np.asarray(constants.type_codes)
    goal = int(_sc(state.goal_token))
    np.testing.assert_allclose(codes[BILLBOARD_IDX], table[goal + 1], atol=1e-6)
    for s in OBJECT_SLOTS:
        t = int(np.asarray(state.circle_types)[s])
        if t >= 0:
            np.testing.assert_allclose(codes[s], table[t + 1], atol=1e-6)
    # Role one-hot block: [pos(2), vel(2), inv_mass, inv_inertia, density,
    # tanh(ang_vel), role_onehot(4), ...] -> columns 8:12. All roles are 0,
    # so column 8 (role 0) is a constant 1 and the green/blue/red indicator
    # columns 9:12 must be identically zero — no role information exists.
    assert np.abs(np.asarray(obs.circles[:, 9:12])).sum() == 0.0
    assert np.abs(np.asarray(obs.polygons[:, 9:12])).sum() == 0.0
    assert np.unique(np.asarray(obs.circles[:, 8])).tolist() == [1.0]
    assert np.unique(np.asarray(obs.polygons[:, 8])).tolist() == [1.0]


def test_auto_reset_on_done(setup):
    env, env_params = setup["env"], setup["env_params"]
    state = _fresh_depth_state(setup, 1)
    goal = int(_sc(state.goal_token))
    types = np.asarray(state.circle_types)
    goal_slot = [s for s in OBJECT_SLOTS if types[s] == goal][0]
    state = _place_in_zone(state, {goal_slot: _zone_center(setup["constants"])})
    _obs, next_state, reward, done, _info = env.step(
        jax.random.PRNGKey(1), state, _zero_action(env), env_params
    )
    assert bool(done) and float(reward) == 1.0
    # auto-reset must give a fresh episode (timestep 0, objects at spawns)
    assert int(_sc(next_state.timestep)) == 0
    pos = np.asarray(next_state.circle.position)[list(OBJECT_SLOTS)]
    spawn = np.asarray(setup["constants"].spawn_positions)
    for p in pos:
        assert min(np.linalg.norm(spawn - p, axis=1)) < 1e-4


def test_event_shaping_bonuses_fire_once_and_are_nonterminal(setup):
    env_shaped = make_banyan_env(
        base_state=setup["base"],
        static_env_params=setup["static"],
        env_params=setup["env_params"],
        constants=setup["constants"],
        reset_fn=setup["reset_fn"],
        action_type="continuous",
        reward_first_lift=0.1,
        reward_first_deposit=0.2,
    )
    ep = setup["env_params"]
    state = _fresh_depth_state(setup, 2)
    lhs, rhs = int(_sc(state.required_lhs)), int(_sc(state.required_rhs))
    types = np.asarray(state.circle_types)
    flags0 = np.asarray(state.lift_flags)
    # A required slot whose first-lift is not pre-consumed (i.e. it did not
    # spawn at the elevated claw spot).
    slot_a = [
        s for s in OBJECT_SLOTS
        if types[s] in (lhs, rhs) and not flags0[list(OBJECT_SLOTS).index(s)]
    ][0]
    lip_top = float(np.asarray(setup["constants"].lip_top))

    # Hold required object A above the lip (mid-air) -> one-time lift bonus.
    circle = state.circle
    circle = circle.replace(
        position=circle.position.at[slot_a].set(jnp.asarray([2.0, lip_top + 0.3])),
        velocity=circle.velocity.at[slot_a].set(0.0),
    )
    state = state.replace(circle=circle)
    _o, state, r1, d1_, _i = env_shaped.step(jax.random.PRNGKey(1), state, _zero_action(env_shaped), ep)
    assert abs(float(r1) - 0.1) < 1e-5 and not bool(d1_)
    # Second step while still lifted: no repeat bonus (object is falling but
    # still above the lip for at least one more step).
    _o, state, r2, d2_, _i = env_shaped.step(jax.random.PRNGKey(2), state, _zero_action(env_shaped), ep)
    assert float(r2) <= 1e-5 and not bool(d2_)

    # Deposit A into the zone alone: deposit bonus, non-terminal (fresh flags
    # for the lift already consumed; deposit not yet).
    state = _place_in_zone(state, {slot_a: _zone_center(setup["constants"])})
    _o, state, r3, d3_, _i = env_shaped.step(jax.random.PRNGKey(3), state, _zero_action(env_shaped), ep)
    assert abs(float(r3) - 0.2) < 1e-5 and not bool(d3_)
    _o, state, r4, d4_, _i = env_shaped.step(jax.random.PRNGKey(4), state, _zero_action(env_shaped), ep)
    assert float(r4) <= 1e-5 and not bool(d4_)

    # Inactive (distractor) slot lifted: NO bonus.
    state2 = _fresh_depth_state(setup, 2)
    distractor = [s for s in OBJECT_SLOTS if not bool(np.asarray(state2.circle.active)[s])][0]
    circle2 = state2.circle
    circle2 = circle2.replace(
        position=circle2.position.at[distractor].set(jnp.asarray([2.0, lip_top + 0.3])),
        velocity=circle2.velocity.at[distractor].set(0.0),
    )
    state2 = state2.replace(circle=circle2)
    _o, _s, r5, d5_, _i = env_shaped.step(jax.random.PRNGKey(5), state2, _zero_action(env_shaped), ep)
    assert float(r5) <= 1e-5 and not bool(d5_)

    # Terminal success still +1 exactly (no bonus double-count on terminal).
    state3 = _fresh_depth_state(setup, 2)
    types3 = np.asarray(state3.circle_types)
    lhs3, rhs3 = int(_sc(state3.required_lhs)), int(_sc(state3.required_rhs))
    sa = [s for s in OBJECT_SLOTS if types3[s] == lhs3][0]
    sb = [s for s in OBJECT_SLOTS if types3[s] == rhs3][0]
    center = _zone_center(setup["constants"])
    state3 = _place_in_zone(
        state3, {sa: center + np.array([-0.15, 0.0]), sb: center + np.array([0.15, 0.0])}
    )
    _o, _s, r6, d6_, info6 = env_shaped.step(jax.random.PRNGKey(6), state3, _zero_action(env_shaped), ep)
    assert abs(float(r6) - 1.0) < 1e-5 and bool(d6_)
    assert bool(np.asarray(info6["GoalR"]))


def test_height_potential_pays_only_new_max_of_required(setup):
    env_h = make_banyan_env(
        base_state=setup["base"],
        static_env_params=setup["static"],
        env_params=setup["env_params"],
        constants=setup["constants"],
        reset_fn=setup["reset_fn"],
        action_type="continuous",
        reward_height_scale=1.0,
    )
    ep = setup["env_params"]
    state = _fresh_depth_state(setup, 2)
    lhs, rhs = int(_sc(state.required_lhs)), int(_sc(state.required_rhs))
    types = np.asarray(state.circle_types)
    flags0 = np.asarray(state.lift_flags)
    slot = [
        s for s in OBJECT_SLOTS
        if types[s] in (lhs, rhs) and not flags0[list(OBJECT_SLOTS).index(s)]
    ][0]
    slot_i = list(OBJECT_SLOTS).index(slot)
    h0 = float(np.asarray(state.max_heights)[slot_i])

    def raise_to(state, y):
        c = state.circle
        c = c.replace(
            position=c.position.at[slot].set(jnp.asarray([2.0, y])),
            velocity=c.velocity.at[slot].set(0.0),
        )
        return state.replace(circle=c)

    # +0.3m new max -> reward ~ 0.3 (gravity pulls it down a hair during the
    # physics substeps, so allow small tolerance).
    state = raise_to(state, h0 + 0.3)
    _o, state, r1, d1_, _i = env_h.step(jax.random.PRNGKey(1), state, _zero_action(env_h), ep)
    assert 0.25 <= float(r1) <= 0.31 and not bool(d1_)
    # Lower it back down: no reward, and no re-payment when re-raised to the
    # SAME height.
    state = raise_to(state, h0 + 0.1)
    _o, state, r2, _d, _i = env_h.step(jax.random.PRNGKey(2), state, _zero_action(env_h), ep)
    assert float(r2) <= 1e-4
    state = raise_to(state, h0 + 0.28)
    _o, state, r3, _d, _i = env_h.step(jax.random.PRNGKey(3), state, _zero_action(env_h), ep)
    assert float(r3) <= 1e-4
    # Inactive (distractor) slot raised: nothing.
    distractor = [s for s in OBJECT_SLOTS if not bool(np.asarray(state.circle.active)[s])][0]
    c = state.circle
    c = c.replace(
        position=c.position.at[distractor].set(jnp.asarray([2.4, h0 + 0.5])),
        velocity=c.velocity.at[distractor].set(0.0),
    )
    state = state.replace(circle=c)
    _o, _s, r4, _d, _i = env_h.step(jax.random.PRNGKey(4), state, _zero_action(env_h), ep)
    assert float(r4) <= 1e-4


def test_vmapped_reset_and_step(setup):
    env, env_params, reset_fn = setup["env"], setup["env_params"], setup["reset_fn"]
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states = jax.vmap(reset_fn)(keys)
    actions = jnp.zeros((8,) + env.action_space(None).shape, dtype=jnp.float32)
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    _obs, next_states, rewards, dones, _info = step(keys, states, actions, env_params)
    assert next_states.circle_types.shape == (8, setup["static"].num_circles)
    assert not np.asarray(dones).any()


def test_num_distractors_activates_extra_bank_slots(setup):
    d1_bank = setup["d1_bank"]
    for k in (1, 3):
        reset_k = make_banyan_reset_fn(setup["base"], setup["constants"], d1_bank, num_distractors=k)
        for seed in range(6):
            state = reset_k(jax.random.PRNGKey(seed))
            row = int(_sc(state.task_id))
            req = np.asarray(d1_bank["required_leaf_mask"])[row]
            active = np.asarray(state.circle.active)[list(OBJECT_SLOTS)]
            types = np.asarray(state.circle_types)[list(OBJECT_SLOTS)]
            bank_tokens = np.asarray(d1_bank["leaf_token_ids"])[row]
            assert (active[req]).all()
            assert int(active.sum()) == int(req.sum()) + min(k, NUM_OBJECT_SLOTS - int(req.sum()))
            # every active slot carries the bank's token for that slot; inactive are untyped
            assert (types[active] == bank_tokens[active]).all()
            assert (types[~active] == -1).all()
            # distractor types are never the goal type
            goal = int(_sc(state.goal_token))
            assert all(types[i] != goal for i in range(NUM_OBJECT_SLOTS) if active[i] and not req[i])


def test_obs_rule_block_carries_tree_info(setup):
    env, constants = setup["env"], setup["constants"]
    table = np.asarray(constants.type_codes)
    D = TYPE_CODE_DIM
    for depth in (1, 2):
        state = _fresh_depth_state(setup, depth)
        obs = env.get_obs(state)
        block = np.asarray(obs.circles[:, 19 + D :])
        assert block.shape[1] == RULE_BLOCK_DIM
        others = np.delete(block, RULE_BILLBOARD_IDX, axis=0)
        assert np.abs(others).sum() == 0.0  # rule info only on the rule billboard row
        row = block[RULE_BILLBOARD_IDX]
        if depth == 1:
            assert np.abs(row).sum() == 0.0  # no rule at depth 1
        else:
            lhs, rhs, out = (int(_sc(state.required_lhs)), int(_sc(state.required_rhs)), int(_sc(state.required_out)))
            assert out == int(_sc(state.goal_token))
            np.testing.assert_allclose(row[:D], table[lhs + 1], atol=1e-6)
            np.testing.assert_allclose(row[D : 2 * D], table[rhs + 1], atol=1e-6)
            np.testing.assert_allclose(row[2 * D : 3 * D], table[out + 1], atol=1e-6)
            assert row[3 * D] == 1.0
    # both billboards fixated and non-colliding
    base = setup["base"]
    assert bool(base.circle.active[RULE_BILLBOARD_IDX])
    assert int(base.circle.collision_mode[RULE_BILLBOARD_IDX]) == 0


def test_wrong_deposit_penalty_depth1(setup):
    # Non-terminal: -p the first time each distractor enters the zone; success still +1 (minus new penalties).
    reset3 = make_banyan_reset_fn(setup["base"], setup["constants"], setup["d1_bank"], num_distractors=3)
    mk = lambda pen: make_banyan_env(
        base_state=setup["base"], static_env_params=setup["static"], env_params=setup["env_params"],
        constants=setup["constants"], reset_fn=reset3, action_type="continuous", reward_wrong_deposit=pen,
    )
    env_pen, env_off = mk(0.3), mk(0.0)
    ep = setup["env_params"]
    bank = setup["d1_bank"]
    idx = int(np.nonzero(np.asarray(bank["depth"]) == 1)[0][0])
    row_bank = {k: v[idx : idx + 1] for k, v in bank.items()}
    state = make_banyan_reset_fn(setup["base"], setup["constants"], row_bank, num_distractors=3)(jax.random.PRNGKey(5))
    goal = int(_sc(state.goal_token))
    types = np.asarray(state.circle_types)
    wrongs = [s for s in OBJECT_SLOTS if types[s] >= 0 and types[s] != goal]
    right = [s for s in OBJECT_SLOTS if types[s] == goal][0]
    center = _zone_center(setup["constants"])
    # one distractor in: -0.3, non-terminal; second step same object: no repeat
    st = _place_in_zone(state, {wrongs[0]: center})
    _o, st, r, d, info = env_pen.step(jax.random.PRNGKey(1), st, _zero_action(env_pen), ep)
    assert abs(float(r) + 0.3) < 1e-5 and not bool(d) and int(np.asarray(info["n_wrong_in_zone"])) == 1
    _o, st, r2, d2, _i = env_pen.step(jax.random.PRNGKey(2), st, _zero_action(env_pen), ep)
    assert float(r2) > -1e-5 and not bool(d2)
    # penalty off: nothing
    _o, _s, r0, d0, _i = env_off.step(jax.random.PRNGKey(1), _place_in_zone(state, {wrongs[0]: center}), _zero_action(env_off), ep)
    assert float(r0) == 0.0 and not bool(d0)
    # bulldoze everything in at once: +1 - 3*0.3 = 0.1, terminal success
    st3 = _place_in_zone(state, {right: center, wrongs[0]: center + np.array([-0.3, 0.0]),
                                 wrongs[1]: center + np.array([0.3, 0.0]), wrongs[2]: center + np.array([0.0, 0.3])})
    _o, _s, r3, d3, info3 = env_pen.step(jax.random.PRNGKey(1), st3, _zero_action(env_pen), ep)
    assert abs(float(r3) - 0.1) < 1e-5 and bool(d3) and bool(np.asarray(info3["GoalR"]))
    # goal alone: exactly +1
    _o, _s, r4, d4, _i = env_pen.step(jax.random.PRNGKey(1), _place_in_zone(state, {right: center}), _zero_action(env_pen), ep)
    assert abs(float(r4) - 1.0) < 1e-5 and bool(d4)
