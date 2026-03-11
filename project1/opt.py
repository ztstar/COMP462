import numpy as np
import copy
import sim
import goal
import rrt
import utils


########## TODO ##########

def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_quat(quat):
    x, y, z, w = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _extract_controls(plan):
    ctrls = []
    for node in plan:
        ctrl = node.get_control()
        if ctrl is not None:
            ctrls.append(np.array(ctrl, dtype=float))
    return ctrls


def _make_plan_from_states_and_controls(states, controls):
    plan = []
    if len(states) == 0:
        return plan

    start_node = rrt.Node(states[0])
    start_node.set_parent(None)
    start_node.set_control(None)
    plan.append(start_node)

    parent = start_node
    for i, ctrl in enumerate(controls):
        node = rrt.Node(states[i + 1])
        node.set_parent(parent)
        node.set_control(np.array(ctrl, dtype=float))
        plan.append(node)
        parent = node

    return plan

def _object_disturbance_cost(prev_state, curr_state):
    prev_vec = prev_state["stateVec"]
    curr_vec = curr_state["stateVec"]

    if prev_vec.shape[0] <= sim.pandaNumDofs:
        return 0.0

    prev_obj = prev_vec[sim.pandaNumDofs:]
    curr_obj = curr_vec[sim.pandaNumDofs:]
    return np.sum((curr_obj - prev_obj) ** 2)


def _control_smoothness_cost(ctrl_prev, ctrl_curr):
    return np.sum((ctrl_curr - ctrl_prev) ** 2)


def _control_effort_cost(ctrl):
    # Prefer shorter/smaller controls
    vmag = np.linalg.norm(ctrl[:3])
    return vmag * ctrl[3]


def rollout_controls(pdef, controls):
    """
    Roll out a control sequence from the problem start state.
    Returns:
        plan: list of rrt.Node
        valid: bool
        total_cost: float
    """
    panda_sim = pdef.panda_sim
    start_state = pdef.get_start_state()

    panda_sim.restore_state(start_state)
    states = [panda_sim.save_state()]
    used_controls = []

    total_cost = 0.0
    prev_ctrl = None
    valid_all = True

    for ctrl in controls:
        panda_sim.restore_state(states[-1])

        _, valid = panda_sim.execute(np.array(ctrl, dtype=float), sleep_time=0.0)
        next_state = panda_sim.save_state()

        if (not valid) or (not pdef.is_state_valid(next_state)):
            valid_all = False
            break

        total_cost += 200.0 * _object_disturbance_cost(states[-1], next_state)
        total_cost += 0.1 * _control_effort_cost(ctrl)

        if prev_ctrl is not None:
            total_cost += 0.5 * _control_smoothness_cost(prev_ctrl, ctrl)

        states.append(next_state)
        used_controls.append(np.array(ctrl, dtype=float))
        prev_ctrl = np.array(ctrl, dtype=float)

        if pdef.get_goal().is_satisfied(next_state):
            # Early stop if goal already achieved
            break

    if not valid_all or not pdef.get_goal().is_satisfied(states[-1]):
        total_cost += 1e6

    plan = _make_plan_from_states_and_controls(states, used_controls)
    return plan, valid_all, total_cost


def compute_plan_cost(pdef, plan):
    ctrls = _extract_controls(plan)
    _, _, cost = rollout_controls(pdef, ctrls)
    return cost


def _perturb_controls(ctrls, low, high, sigma):
    new_ctrls = []
    for ctrl in ctrls:
        c = np.array(ctrl, dtype=float).copy()
        noise = np.random.normal(loc=0.0, scale=sigma, size=c.shape)
        c = c + noise
        c = np.clip(c, low, high)
        new_ctrls.append(c)
    return new_ctrls


def _greedy_time_shorten(ctrls, low, high):
    """
    Small deterministic refinement:
    try to reduce duration a bit while staying in bounds.
    """
    out = []
    for c in ctrls:
        cc = np.array(c, dtype=float).copy()
        cc[3] = max(low[3], cc[3] - 0.03)
        cc = np.clip(cc, low, high)
        out.append(cc)
    return out


def optimize_plan(pdef, init_plan, iterations=25, num_rollouts=8, sigma=None):
    """
    Stochastic trajectory optimization using the RRT plan as initialization.
    """
    if sigma is None:
        sigma = np.array([0.02, 0.02, 0.15, 0.05], dtype=float)

    low = pdef.bounds_ctrl.low
    high = pdef.bounds_ctrl.high

    best_ctrls = _extract_controls(init_plan)
    best_plan, best_valid, best_cost = rollout_controls(pdef, best_ctrls)

    if len(best_ctrls) == 0:
        return init_plan, {"best_cost": best_cost, "iterations": 0}

    for _ in range(iterations):
        candidate_ctrls = []

        # keep current best
        candidate_ctrls.append([c.copy() for c in best_ctrls])

        # add a simple shortening candidate
        candidate_ctrls.append(_greedy_time_shorten(best_ctrls, low, high))

        # stochastic rollouts
        for _ in range(num_rollouts):
            candidate_ctrls.append(_perturb_controls(best_ctrls, low, high, sigma))

        improved = False
        local_best_ctrls = best_ctrls
        local_best_plan = best_plan
        local_best_cost = best_cost

        for ctrls in candidate_ctrls:
            cand_plan, cand_valid, cand_cost = rollout_controls(pdef, ctrls)

            # accept only valid rollouts
            if cand_valid and cand_cost < local_best_cost:
                local_best_ctrls = ctrls
                local_best_plan = cand_plan
                local_best_cost = cand_cost
                improved = True

        if improved:
            best_ctrls = [c.copy() for c in local_best_ctrls]
            best_plan = local_best_plan
            best_cost = local_best_cost

    return best_plan, {"best_cost": best_cost, "iterations": iterations}


##########################
