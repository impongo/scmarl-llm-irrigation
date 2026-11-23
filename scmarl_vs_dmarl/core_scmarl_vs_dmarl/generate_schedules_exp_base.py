import os
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from tensorforce.agents import Agent

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(MODULE_DIR, "..", "..", "common", "environments")))
sys.path.append(os.path.abspath(os.path.join(MODULE_DIR, "..", "..", "common", "actual_field")))
from environments import SimulateModel_LA_a, SimulateModel_LA_b, SimulateModel_LA_c, SimulateModel_UA
from temporal_spatial_scheduler import pars_mz1, pars_mz2, pars_mz3, spatialVariables_1D
from van_Genuchten_scheduler import volumetricMoisture

totalDepth, axialNodes, totalNodes = spatialVariables_1D()

no_of_mzs = 3

soilPars_mz1 = pars_mz1()
soilPars_mz2 = pars_mz2()
soilPars_mz3 = pars_mz3()

LOWER_BOUND_MZ1 = 0.200
LOWER_BOUND_MZ2 = 0.200
LOWER_BOUND_MZ3 = 0.230
LOWER_BOUNDS = np.array([LOWER_BOUND_MZ1, LOWER_BOUND_MZ2, LOWER_BOUND_MZ3], dtype=float)

LOCAL_AGENT_PROFILES = {
    "scmarl_local": {
        "agent_dir": "./trained_rl_agents_a/agents_scmarl/",
        "agent_filenames": {
            "mz1": "Scheduler_agent_b - 12-2",
            "mz2": "Scheduler_agent_c - 12-2",
            "mz3": "Scheduler_agent_d - 12-2",
        },
        "append_allow_to_state": True,
    },
    "dmarl_local": {
        "agent_dir": "./trained_rl_agents_a/agents_dmarl/",
        "agent_filenames": {
            "mz1": "Scheduler_agent_b - 14-4",
            "mz2": "Scheduler_agent_c - 14-4",
            "mz3": "Scheduler_agent_d - 14-4",
        },
        "append_allow_to_state": False,
    },
}

_AGENT_CACHE: Dict[Tuple[str, str, str, str], Tuple[Agent, Agent, Agent]] = {}


def _compute_root_zone_moisture(axial_state: np.ndarray) -> float:
    return (
        0.10 * (1 / 6) * (axial_state[0] + axial_state[1] + axial_state[2] + axial_state[3] + axial_state[4] + axial_state[5])
        + 0.20 * (1 / 6) * (axial_state[5] + axial_state[6] + axial_state[7] + axial_state[8] + axial_state[9] + axial_state[10])
        + 0.30 * (1 / 6) * (axial_state[10] + axial_state[11] + axial_state[12] + axial_state[13] + axial_state[14] + axial_state[15])
        + 0.40 * (1 / 6) * (axial_state[15] + axial_state[16] + axial_state[17] + axial_state[18] + axial_state[19] + axial_state[20])
    )


def _rule_global_allow(
    state_mz1: np.ndarray,
    state_mz2: np.ndarray,
    state_mz3: np.ndarray,
    threshold_margin: float = 0.0,
) -> float:
    rz_moisture = np.array(
        [
            _compute_root_zone_moisture(state_mz1),
            _compute_root_zone_moisture(state_mz2),
            _compute_root_zone_moisture(state_mz3),
        ],
        dtype=float,
    )
    return 1.0 if np.any(rz_moisture < (LOWER_BOUNDS + threshold_margin)) else 0.0


def _rule_per_zone_allow(
    state_mz1: np.ndarray,
    state_mz2: np.ndarray,
    state_mz3: np.ndarray,
    threshold_margin: float = 0.0,
) -> np.ndarray:
    rz_moisture = np.array(
        [
            _compute_root_zone_moisture(state_mz1),
            _compute_root_zone_moisture(state_mz2),
            _compute_root_zone_moisture(state_mz3),
        ],
        dtype=float,
    )
    return (rz_moisture < (LOWER_BOUNDS + threshold_margin)).astype(float)


def _to_scalar_action(action_value: Any) -> float:
    arr = np.asarray(action_value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _clip_continuous_action(action_value: float) -> float:
    if abs(action_value) > 0.040:
        return -0.040
    return action_value


def _normalize_binary(value: Any) -> float:
    return 1.0 if float(value) >= 0.5 else 0.0


def _normalize_binary_vector(values: Iterable[Any]) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float).reshape(-1)
    if values_array.size != 3:
        raise ValueError("per-zone decision must contain exactly 3 values")
    return (values_array >= 0.5).astype(float)


def _build_step_context(
    step_index: int,
    prediction_horizon: int,
    state_mz1: np.ndarray,
    state_mz2: np.ndarray,
    state_mz3: np.ndarray,
    uncontrolled_inputs: np.ndarray,
) -> Dict[str, Any]:
    rz_moisture = [
        _compute_root_zone_moisture(state_mz1),
        _compute_root_zone_moisture(state_mz2),
        _compute_root_zone_moisture(state_mz3),
    ]
    return {
        "step_index": int(step_index),
        "prediction_horizon": int(prediction_horizon),
        "root_zone_moisture": rz_moisture,
        "lower_bounds": LOWER_BOUNDS.tolist(),
        "current_inputs": {
            "ref_et_mm_per_day": float(uncontrolled_inputs[step_index, 0]),
            "crop_coeff": float(uncontrolled_inputs[step_index, 1]),
            "lai": float(uncontrolled_inputs[step_index, 2]),
            "rain_mm_per_day": float(uncontrolled_inputs[step_index, 3] * 1000.0),
        },
        "future_inputs": {
            "ref_et_mm_per_day": uncontrolled_inputs[step_index:, 0].astype(float).tolist(),
            "crop_coeff": uncontrolled_inputs[step_index:, 1].astype(float).tolist(),
            "lai": uncontrolled_inputs[step_index:, 2].astype(float).tolist(),
            "rain_mm_per_day": (uncontrolled_inputs[step_index:, 3].astype(float) * 1000.0).tolist(),
        },
    }


def _get_local_agents(local_agent_profile: str) -> Tuple[Agent, Agent, Agent, bool]:
    if local_agent_profile not in LOCAL_AGENT_PROFILES:
        raise ValueError(f"Unknown local_agent_profile: {local_agent_profile}")

    profile = LOCAL_AGENT_PROFILES[local_agent_profile]
    agent_dir = profile["agent_dir"]
    agent_filenames = profile["agent_filenames"]

    key = (
        agent_dir,
        agent_filenames["mz1"],
        agent_filenames["mz2"],
        agent_filenames["mz3"],
    )
    if key not in _AGENT_CACHE:
        agent_b = Agent.load(agent_dir, filename=agent_filenames["mz1"])
        agent_c = Agent.load(agent_dir, filename=agent_filenames["mz2"])
        agent_d = Agent.load(agent_dir, filename=agent_filenames["mz3"])
        _AGENT_CACHE[key] = (agent_b, agent_c, agent_d)

    append_allow_to_state = bool(profile["append_allow_to_state"])
    return _AGENT_CACHE[key][0], _AGENT_CACHE[key][1], _AGENT_CACHE[key][2], append_allow_to_state


def evaluate_scheduler_head_generic(
    states_mz1: np.ndarray,
    states_mz2: np.ndarray,
    states_mz3: np.ndarray,
    refEvap_sequence: np.ndarray,
    cropCoeff_sequence: np.ndarray,
    lai_sequence: np.ndarray,
    rain_sequence: np.ndarray,
    prediction_horizon: int,
    coordinator_mode: str,
    local_agent_profile: str,
    llm_callback: Optional[Callable[[Dict[str, Any]], Optional[Any]]] = None,
    llm_call_policy: str = "per_step",
    rule_threshold_margin: float = 0.0,
):
    if llm_call_policy not in ("per_step", "once_per_day"):
        raise ValueError(f"Unsupported llm_call_policy: {llm_call_policy}")

    states_init_volMoist_mz1 = volumetricMoisture(states_mz1, soilPars_mz1)[10:]
    states_init_volMoist_mz2 = volumetricMoisture(states_mz2, soilPars_mz2)[10:]
    states_init_volMoist_mz3 = volumetricMoisture(states_mz3, soilPars_mz3)[10:]

    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:, 0] = refEvap_sequence * (86400 * 1000)
    uncontrolled_inputs[:, 1] = cropCoeff_sequence
    uncontrolled_inputs[:, 2] = lai_sequence
    uncontrolled_inputs[:, 3] = np.abs(rain_sequence)

    agent_b, agent_c, agent_d, append_allow_to_state = _get_local_agents(local_agent_profile)
    internals_b = agent_b.initial_internals()
    internals_c = agent_c.initial_internals()
    internals_d = agent_d.initial_internals()

    stateArray_ua = np.zeros((prediction_horizon + 1, int(3 * axialNodes)))
    stateArray_b = np.zeros((prediction_horizon + 1, axialNodes))
    stateArray_c = np.zeros((prediction_horizon + 1, axialNodes))
    stateArray_d = np.zeros((prediction_horizon + 1, axialNodes))

    stateArray_rz_b = np.zeros(prediction_horizon + 1)
    stateArray_rz_c = np.zeros(prediction_horizon + 1)
    stateArray_rz_d = np.zeros(prediction_horizon + 1)

    actionsArray_b = np.zeros(prediction_horizon)
    actionsArray_c = np.zeros(prediction_horizon)
    actionsArray_d = np.zeros(prediction_horizon)

    decision_width = 1 if coordinator_mode in ("rule_sync", "llm_sync") else 3
    decision_log = np.zeros((prediction_horizon, decision_width))

    vol_init_a = np.zeros(3 * axialNodes)
    vol_init_a[0 * axialNodes : 1 * axialNodes] = states_init_volMoist_mz1
    vol_init_a[1 * axialNodes : 2 * axialNodes] = states_init_volMoist_mz2
    vol_init_a[2 * axialNodes : 3 * axialNodes] = states_init_volMoist_mz3

    stateArray_ua[0, :] = vol_init_a
    stateArray_b[0, :] = states_init_volMoist_mz1
    stateArray_c[0, :] = states_init_volMoist_mz2
    stateArray_d[0, :] = states_init_volMoist_mz3

    stateArray_rz_b[0] = _compute_root_zone_moisture(states_init_volMoist_mz1)
    stateArray_rz_c[0] = _compute_root_zone_moisture(states_init_volMoist_mz2)
    stateArray_rz_d[0] = _compute_root_zone_moisture(states_init_volMoist_mz3)

    cached_sync_decision: Optional[float] = None
    cached_per_zone_decision: Optional[np.ndarray] = None

    for i in range(prediction_horizon):
        states_ua = np.concatenate((stateArray_ua[i, :], uncontrolled_inputs[i, :]))
        states_b = np.concatenate((stateArray_b[i, :], uncontrolled_inputs[i, :]))
        states_c = np.concatenate((stateArray_c[i, :], uncontrolled_inputs[i, :]))
        states_d = np.concatenate((stateArray_d[i, :], uncontrolled_inputs[i, :]))

        current_state_ua = states_ua
        current_state_b = states_b
        current_state_c = states_c
        current_state_d = states_d
        rain_mm = float(abs(rain_sequence[i]) * 1000.0)

        if rain_mm > 0.0:
            allow_vector = np.zeros(no_of_mzs)
            global_allow = 0.0
            action_b = 0.0
            action_c = 0.0
            action_d = 0.0
        else:
            context = _build_step_context(
                step_index=i,
                prediction_horizon=prediction_horizon,
                state_mz1=stateArray_b[i, :],
                state_mz2=stateArray_c[i, :],
                state_mz3=stateArray_d[i, :],
                uncontrolled_inputs=uncontrolled_inputs,
            )

            if coordinator_mode == "rule_sync":
                global_allow = _rule_global_allow(
                    stateArray_b[i, :],
                    stateArray_c[i, :],
                    stateArray_d[i, :],
                    threshold_margin=rule_threshold_margin,
                )
                allow_vector = np.array([global_allow, global_allow, global_allow], dtype=float)
            elif coordinator_mode == "llm_sync":
                if llm_call_policy == "once_per_day":
                    if cached_sync_decision is None:
                        llm_decision = llm_callback(context) if llm_callback else None
                        cached_sync_decision = _rule_global_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :]) if llm_decision is None else _normalize_binary(llm_decision)
                    global_allow = cached_sync_decision
                else:
                    llm_decision = llm_callback(context) if llm_callback else None
                    global_allow = _rule_global_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :]) if llm_decision is None else _normalize_binary(llm_decision)
                allow_vector = np.array([global_allow, global_allow, global_allow], dtype=float)
            elif coordinator_mode == "llm_per_zone":
                if llm_call_policy == "once_per_day":
                    if cached_per_zone_decision is None:
                        llm_decision = llm_callback(context) if llm_callback else None
                        cached_per_zone_decision = _rule_per_zone_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :]) if llm_decision is None else _normalize_binary_vector(llm_decision)
                    allow_vector = cached_per_zone_decision
                else:
                    llm_decision = llm_callback(context) if llm_callback else None
                    allow_vector = _rule_per_zone_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :]) if llm_decision is None else _normalize_binary_vector(llm_decision)
                global_allow = float(np.max(allow_vector))
            else:
                raise ValueError(f"Unsupported coordinator_mode: {coordinator_mode}")

            if append_allow_to_state:
                states_b_for_agent = np.concatenate((states_b, np.array([allow_vector[0]])))
                states_c_for_agent = np.concatenate((states_c, np.array([allow_vector[1]])))
                states_d_for_agent = np.concatenate((states_d, np.array([allow_vector[2]])))
            else:
                states_b_for_agent = states_b
                states_c_for_agent = states_c
                states_d_for_agent = states_d

            raw_action_b, internals_b = agent_b.act(states=states_b_for_agent, internals=internals_b, independent=True)
            raw_action_c, internals_c = agent_c.act(states=states_c_for_agent, internals=internals_c, independent=True)
            raw_action_d, internals_d = agent_d.act(states=states_d_for_agent, internals=internals_d, independent=True)

            action_b = _clip_continuous_action(_to_scalar_action(raw_action_b)) * allow_vector[0]
            action_c = _clip_continuous_action(_to_scalar_action(raw_action_c)) * allow_vector[1]
            action_d = _clip_continuous_action(_to_scalar_action(raw_action_d)) * allow_vector[2]

        action_all = np.zeros(no_of_mzs)
        action_all[0] = action_b - uncontrolled_inputs[i, 3]
        action_all[1] = action_c - uncontrolled_inputs[i, 3]
        action_all[2] = action_d - uncontrolled_inputs[i, 3]

        next_state_ua = SimulateModel_UA(current_state_ua[0 : 3 * axialNodes], action_all, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
        next_state_b = SimulateModel_LA_a(current_state_b[0:axialNodes], action_all[0], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
        next_state_c = SimulateModel_LA_b(current_state_c[0:axialNodes], action_all[1], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
        next_state_d = SimulateModel_LA_c(current_state_d[0:axialNodes], action_all[2], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)

        actionsArray_b[i] = action_b
        actionsArray_c[i] = action_c
        actionsArray_d[i] = action_d

        if decision_width == 1:
            decision_log[i, 0] = global_allow
        else:
            decision_log[i, :] = allow_vector

        stateArray_ua[i + 1, :] = next_state_ua
        stateArray_b[i + 1, :] = next_state_b
        stateArray_c[i + 1, :] = next_state_c
        stateArray_d[i + 1, :] = next_state_d

        stateArray_rz_b[i + 1] = _compute_root_zone_moisture(next_state_b)
        stateArray_rz_c[i + 1] = _compute_root_zone_moisture(next_state_c)
        stateArray_rz_d[i + 1] = _compute_root_zone_moisture(next_state_d)

    decisions = decision_log[:, 0] if decision_width == 1 else decision_log
    return stateArray_rz_b, stateArray_rz_c, stateArray_rz_d, decisions, actionsArray_b, actionsArray_c, actionsArray_d
