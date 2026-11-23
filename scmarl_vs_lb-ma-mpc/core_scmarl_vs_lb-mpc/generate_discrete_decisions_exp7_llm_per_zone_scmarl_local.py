import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
from tensorforce.agents import Agent

sys.path.append('../common/environments/')
from environments import SimulateModel_UA, SimulateModel_LA_a, SimulateModel_LA_b, SimulateModel_LA_c
from temporal_spatial_scheduler import spatialVariables_1D, pars_mz1, pars_mz2, pars_mz3
from van_Genuchten_scheduler import volumetricMoisture

totalDepth, axialNodes, totalNodes = spatialVariables_1D()

no_of_mzs = 3
soilPars_mz1 = pars_mz1()
soilPars_mz2 = pars_mz2()
soilPars_mz3 = pars_mz3()

LOWER_BOUNDS = np.array([0.200, 0.200, 0.230], dtype=float)

_LLM_CALLBACK = None
_LLM_CALL_POLICY = 'once_per_day'
_FALLBACK_WARNED = False


def configure_llm(callback=None, call_policy='once_per_day'):
    global _LLM_CALLBACK, _LLM_CALL_POLICY
    _LLM_CALLBACK = callback
    _LLM_CALL_POLICY = call_policy


def compute_root_zone_moisture(axial_state):
    root_zone_vol_moist = (
        0.10 * ((1 / 6) * (axial_state[0] + axial_state[1] + axial_state[2] + axial_state[3] + axial_state[4] + axial_state[5]))
        + 0.20 * ((1 / 6) * (axial_state[5] + axial_state[6] + axial_state[7] + axial_state[8] + axial_state[9] + axial_state[10]))
        + 0.30 * ((1 / 6) * (axial_state[10] + axial_state[11] + axial_state[12] + axial_state[13] + axial_state[14] + axial_state[15]))
        + 0.40 * ((1 / 6) * (axial_state[15] + axial_state[16] + axial_state[17] + axial_state[18] + axial_state[19] + axial_state[20]))
    )
    return root_zone_vol_moist


def _rule_per_zone_allow(state_mz1, state_mz2, state_mz3):
    rz = np.array(
        [
            compute_root_zone_moisture(state_mz1),
            compute_root_zone_moisture(state_mz2),
            compute_root_zone_moisture(state_mz3),
        ],
        dtype=float,
    )
    return (rz < LOWER_BOUNDS).astype(float)


def _normalize_binary_vector(values: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(list(values), dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size != 3:
        return None
    return (arr >= 0.5).astype(float)


def _build_step_context(step_index: int, prediction_horizon: int, state_mz1, state_mz2, state_mz3, uncontrolled_inputs):
    rz = [
        compute_root_zone_moisture(state_mz1),
        compute_root_zone_moisture(state_mz2),
        compute_root_zone_moisture(state_mz3),
    ]
    return {
        'step_index': int(step_index),
        'prediction_horizon': int(prediction_horizon),
        'root_zone_moisture': rz,
        'lower_bounds': LOWER_BOUNDS.tolist(),
        'current_inputs': {
            'ref_et_mm_per_day': float(uncontrolled_inputs[step_index, 0]),
            'crop_coeff': float(uncontrolled_inputs[step_index, 1]),
            'lai': float(uncontrolled_inputs[step_index, 2]),
            'rain_mm_per_day': float(uncontrolled_inputs[step_index, 3] * 1000.0),
        },
        'future_inputs': {
            'ref_et_mm_per_day': uncontrolled_inputs[step_index:, 0].astype(float).tolist(),
            'crop_coeff': uncontrolled_inputs[step_index:, 1].astype(float).tolist(),
            'lai': uncontrolled_inputs[step_index:, 2].astype(float).tolist(),
            'rain_mm_per_day': (uncontrolled_inputs[step_index:, 3].astype(float) * 1000.0).tolist(),
        },
    }


def evaluate_scheduler_head(states_mz1, states_mz2, states_mz3, refEvap_sequence, cropCoeff_sequence, lai_sequence, rain_sequence, prediction_horizon):
    global _FALLBACK_WARNED

    if _LLM_CALL_POLICY not in ('once_per_day', 'per_step'):
        raise ValueError(f'Unsupported llm call policy: {_LLM_CALL_POLICY}')

    states_init_vol_moist_mz1 = volumetricMoisture(states_mz1, soilPars_mz1)
    states_init_vol_moist_mz2 = volumetricMoisture(states_mz2, soilPars_mz2)
    states_init_vol_moist_mz3 = volumetricMoisture(states_mz3, soilPars_mz3)

    states_init_vol_moist_mz1 = states_init_vol_moist_mz1[10:]
    states_init_vol_moist_mz2 = states_init_vol_moist_mz2[10:]
    states_init_vol_moist_mz3 = states_init_vol_moist_mz3[10:]

    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:, 0] = refEvap_sequence * (86400 * 1000)
    uncontrolled_inputs[:, 1] = cropCoeff_sequence
    uncontrolled_inputs[:, 2] = lai_sequence
    uncontrolled_inputs[:, 3] = np.abs(rain_sequence)

    agent_name_b = 'Scheduler_agent_b - 12-2'
    agent_loaded_b = Agent.load('./trained_rl_agents_b/scmarl_agents/agents/', filename=agent_name_b)
    internals_b = agent_loaded_b.initial_internals()

    agent_name_c = 'Scheduler_agent_c - 12-2'
    agent_loaded_c = Agent.load('./trained_rl_agents_b/scmarl_agents/agents/', filename=agent_name_c)
    internals_c = agent_loaded_c.initial_internals()

    agent_name_d = 'Scheduler_agent_d - 12-2'
    agent_loaded_d = Agent.load('./trained_rl_agents_b/scmarl_agents/agents/', filename=agent_name_d)
    internals_d = agent_loaded_d.initial_internals()

    stateArray_ua = np.zeros((prediction_horizon + 1, int(3 * axialNodes)))
    actionsArray_ua = np.zeros(prediction_horizon)
    gateArray_mz = np.zeros((prediction_horizon, 3))

    stateArray_b = np.zeros((prediction_horizon + 1, axialNodes))
    stateArray_rz_b = np.zeros(prediction_horizon + 1)
    actionsArray_b = np.zeros(prediction_horizon)

    stateArray_c = np.zeros((prediction_horizon + 1, axialNodes))
    stateArray_rz_c = np.zeros(prediction_horizon + 1)
    actionsArray_c = np.zeros(prediction_horizon)

    stateArray_d = np.zeros((prediction_horizon + 1, axialNodes))
    stateArray_rz_d = np.zeros(prediction_horizon + 1)
    actionsArray_d = np.zeros(prediction_horizon)

    vol_init_a = np.zeros(3 * axialNodes)
    vol_init_a[0 * axialNodes : 1 * axialNodes] = states_init_vol_moist_mz1
    vol_init_a[1 * axialNodes : 2 * axialNodes] = states_init_vol_moist_mz2
    vol_init_a[2 * axialNodes : 3 * axialNodes] = states_init_vol_moist_mz3

    stateArray_ua[0, :] = vol_init_a
    stateArray_b[0, :] = states_init_vol_moist_mz1
    stateArray_c[0, :] = states_init_vol_moist_mz2
    stateArray_d[0, :] = states_init_vol_moist_mz3

    stateArray_rz_b[0] = compute_root_zone_moisture(states_init_vol_moist_mz1)
    stateArray_rz_c[0] = compute_root_zone_moisture(states_init_vol_moist_mz2)
    stateArray_rz_d[0] = compute_root_zone_moisture(states_init_vol_moist_mz3)

    cached_allow_vector = None

    for i in range(prediction_horizon):
        states_ua = np.concatenate((stateArray_ua[i, :], uncontrolled_inputs[i, :]))
        states_b = np.concatenate((stateArray_b[i, :], uncontrolled_inputs[i, :]))
        states_c = np.concatenate((stateArray_c[i, :], uncontrolled_inputs[i, :]))
        states_d = np.concatenate((stateArray_d[i, :], uncontrolled_inputs[i, :]))

        if (abs(rain_sequence[i]) * 1000) > 0.00:
            allow_vector = np.zeros(3)
            action_ua = 0.0
            action_b = 0.0
            action_c = 0.0
            action_d = 0.0

            current_state_ua = states_ua
            current_state_b = states_b
            current_state_c = states_c
            current_state_d = states_d

            action_all = np.zeros(no_of_mzs)
            action_all[0] = -uncontrolled_inputs[i, 3]
            action_all[1] = -uncontrolled_inputs[i, 3]
            action_all[2] = -uncontrolled_inputs[i, 3]

            next_state_ua = SimulateModel_UA(current_state_ua[0 : 3 * axialNodes], action_all, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_b = SimulateModel_LA_a(current_state_b[0:axialNodes], -uncontrolled_inputs[i, 3], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_c = SimulateModel_LA_b(current_state_c[0:axialNodes], -uncontrolled_inputs[i, 3], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_d = SimulateModel_LA_c(current_state_d[0:axialNodes], -uncontrolled_inputs[i, 3], uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
        else:
            if _LLM_CALL_POLICY == 'once_per_day':
                if cached_allow_vector is None:
                    context = _build_step_context(i, prediction_horizon, stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :], uncontrolled_inputs)
                    llm_decision = _LLM_CALLBACK(context) if _LLM_CALLBACK else None
                    normalized = _normalize_binary_vector(llm_decision)
                    if normalized is None:
                        normalized = _rule_per_zone_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :])
                        if not _FALLBACK_WARNED:
                            print('[WARN] Exp7 CL-SCMARL did not receive a valid LLM decision. Falling back to rule coordinator.')
                            _FALLBACK_WARNED = True
                    cached_allow_vector = normalized
                allow_vector = cached_allow_vector
            else:
                context = _build_step_context(i, prediction_horizon, stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :], uncontrolled_inputs)
                llm_decision = _LLM_CALLBACK(context) if _LLM_CALLBACK else None
                normalized = _normalize_binary_vector(llm_decision)
                if normalized is None:
                    normalized = _rule_per_zone_allow(stateArray_b[i, :], stateArray_c[i, :], stateArray_d[i, :])
                    if not _FALLBACK_WARNED:
                        print('[WARN] Exp7 CL-SCMARL did not receive a valid LLM decision. Falling back to rule coordinator.')
                        _FALLBACK_WARNED = True
                allow_vector = normalized

            action_ua = float(np.max(allow_vector))

            current_state_ua = states_ua
            current_state_b = states_b
            current_state_c = states_c
            current_state_d = states_d

            states_b = np.concatenate((states_b, np.array([allow_vector[0]])))
            states_c = np.concatenate((states_c, np.array([allow_vector[1]])))
            states_d = np.concatenate((states_d, np.array([allow_vector[2]])))

            action_b, internals_b = agent_loaded_b.act(states=states_b, internals=internals_b, independent=True)
            action_c, internals_c = agent_loaded_c.act(states=states_c, internals=internals_c, independent=True)
            action_d, internals_d = agent_loaded_d.act(states=states_d, internals=internals_d, independent=True)

            if abs(action_b) > 0.040:
                action_b = -0.040
            if abs(action_c) > 0.040:
                action_c = -0.040
            if abs(action_d) > 0.040:
                action_d = -0.040

            action_b = allow_vector[0] * action_b
            action_c = allow_vector[1] * action_c
            action_d = allow_vector[2] * action_d

            action_all = np.zeros(no_of_mzs)
            action_all[0] = action_b - uncontrolled_inputs[i, 3]
            action_all[1] = action_c - uncontrolled_inputs[i, 3]
            action_all[2] = action_d - uncontrolled_inputs[i, 3]

            action_b_mod = action_b - uncontrolled_inputs[i, 3]
            action_c_mod = action_c - uncontrolled_inputs[i, 3]
            action_d_mod = action_d - uncontrolled_inputs[i, 3]

            next_state_ua = SimulateModel_UA(current_state_ua[0 : 3 * axialNodes], action_all, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_b = SimulateModel_LA_a(current_state_b[0:axialNodes], action_b_mod, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_c = SimulateModel_LA_b(current_state_c[0:axialNodes], action_c_mod, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)
            next_state_d = SimulateModel_LA_c(current_state_d[0:axialNodes], action_d_mod, uncontrolled_inputs[i, 0], uncontrolled_inputs[i, 1], uncontrolled_inputs[i, 2], 0)

            actionsArray_ua[i] = action_ua
            actionsArray_b[i] = action_b
            actionsArray_c[i] = action_c
            actionsArray_d[i] = action_d

        gateArray_mz[i, :] = allow_vector

        stateArray_ua[i + 1, :] = next_state_ua
        stateArray_b[i + 1, :] = next_state_b
        stateArray_c[i + 1, :] = next_state_c
        stateArray_d[i + 1, :] = next_state_d

        stateArray_rz_b[i + 1] = compute_root_zone_moisture(next_state_b)
        stateArray_rz_c[i + 1] = compute_root_zone_moisture(next_state_c)
        stateArray_rz_d[i + 1] = compute_root_zone_moisture(next_state_d)

    return stateArray_rz_b, stateArray_rz_c, stateArray_rz_d, gateArray_mz, actionsArray_b, actionsArray_c, actionsArray_d
