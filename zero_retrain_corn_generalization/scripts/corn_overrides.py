import os
import sys
from typing import Dict

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PERF_DIR = os.path.dirname(ROOT_DIR)
COMMON_ACTUAL_FIELD_DIR = os.path.join(PERF_DIR, "common", "actual_field")

if COMMON_ACTUAL_FIELD_DIR not in sys.path:
    sys.path.append(COMMON_ACTUAL_FIELD_DIR)

from van_Genuchten_relations import volumetricMoisture


CORN_PARAMS: Dict = {
    "base_temp_c": 10.0,
    "season_days": 123,
    "stage_days": {"initial": 20, "development": 35, "mid": 38, "late": 30},
    "kc_curve": {
        "kc_ini": 0.30,
        "kc_dev_target": 0.80,
        "kc_mid": 1.20,
        "kc_late": 0.85,
        "kc_harvest": 0.55,
    },
    "yield_model": {
        "maximal_potential_yield": 8.8,
        "yield_response_factor": 1.25,
    },
}


def _load_hydraulic_table() -> np.ndarray:
    path = os.path.join(PERF_DIR, "common", "hydraulic_parameters_mzs.txt")
    return np.loadtxt(path)


def _pars_mz1() -> Dict:
    table = _load_hydraulic_table()
    soil = {}
    soil["thetaR"] = 0.078
    soil["thetaS"] = 0.43
    soil["alpha"] = round(float(table[0, 2]), 1)
    soil["n"] = 1.56
    soil["m"] = 1 - 1 / round(soil["n"], 2)
    soil["Ks"] = round(float(table[0, 0]), 2) / 86400.0
    soil["neta"] = 0.5
    soil["Ss"] = 0.00001
    return soil


def _pars_mz2() -> Dict:
    table = _load_hydraulic_table()
    soil = {}
    soil["thetaR"] = 0.078
    soil["thetaS"] = 0.43
    soil["alpha"] = round(float(table[1, 2]), 1)
    soil["n"] = 1.56
    soil["m"] = 1 - 1 / round(soil["n"], 2)
    soil["Ks"] = round(float(table[1, 0]), 2) / 86400.0
    soil["neta"] = 0.5
    soil["Ss"] = 0.00001
    return soil


def _pars_mz3() -> Dict:
    table = _load_hydraulic_table()
    soil = {}
    soil["thetaR"] = 0.10
    soil["thetaS"] = 0.39
    soil["alpha"] = round(float(table[2, 2]), 1)
    soil["n"] = round(float(table[2, 3]), 2)
    soil["m"] = 1 - 1 / round(soil["n"], 2)
    soil["Ks"] = round(float(table[2, 0]), 2) / 86400.0
    soil["neta"] = 0.5
    soil["Ss"] = 0.00001
    return soil


def _kc_curve_for_day(day_index: int, params: Dict) -> float:
    stage = params["stage_days"]
    kc = params["kc_curve"]
    d1 = stage["initial"]
    d2 = stage["development"]
    d3 = stage["mid"]
    d4 = stage["late"]

    t = day_index + 1
    if t <= d1:
        return kc["kc_ini"]

    if t <= d1 + d2:
        within = t - d1
        part_a = max(1, int(round(0.60 * d2)))
        part_b = max(1, d2 - part_a)
        if within <= part_a:
            frac = within / float(part_a)
            return kc["kc_ini"] + frac * (kc["kc_dev_target"] - kc["kc_ini"])
        frac = (within - part_a) / float(part_b)
        return kc["kc_dev_target"] + frac * (kc["kc_mid"] - kc["kc_dev_target"])

    if t <= d1 + d2 + d3:
        return kc["kc_mid"]

    if t <= d1 + d2 + d3 + d4:
        within = t - (d1 + d2 + d3)
        frac = within / float(max(1, d4))
        return kc["kc_late"] + frac * (kc["kc_harvest"] - kc["kc_late"])

    return kc["kc_harvest"]


def generate_kc_maize(daily_average_temps: np.ndarray, params: Dict = CORN_PARAMS) -> np.ndarray:
    n = len(daily_average_temps)
    kc = np.zeros(n, dtype=float)
    for i in range(n):
        kc[i] = _kc_curve_for_day(i, params)
    return np.clip(kc, 0.20, 1.25)


def _water_stress_factor(theta: float, pwp: float, lower_zone: float, upper_zone: float, anaerobic: float) -> float:
    if theta <= pwp:
        return 0.0
    if pwp < theta <= lower_zone:
        return (theta - pwp) / (lower_zone - pwp)
    if lower_zone < theta < upper_zone:
        return 1.0
    if upper_zone <= theta <= anaerobic:
        return (theta - upper_zone) / (anaerobic - upper_zone)
    return 0.0


def evaluate_yield_maize_from_state_trajectories(
    state_traj_mz1: np.ndarray,
    state_traj_mz2: np.ndarray,
    state_traj_mz3: np.ndarray,
    params: Dict = CORN_PARAMS,
) -> Dict[str, float]:
    weather_dir = os.path.join(PERF_DIR, "common", "weather_data")
    avg_temp = np.loadtxt(os.path.join(weather_dir, "dAVG_TEMP.txt"))
    ref_evap = np.loadtxt(os.path.join(weather_dir, "dPET.txt"))
    ref_evap = np.clip(ref_evap, 1.04, 9.0)
    crop_coeff = generate_kc_maize(avg_temp, params=params)

    lower_mz = [0.200, 0.200, 0.230]
    upper_mz = [0.281, 0.281, 0.301]
    soil_1 = _pars_mz1()
    soil_2 = _pars_mz2()
    soil_3 = _pars_mz3()
    pwp = [
        volumetricMoisture(-152.85, soil_1),
        volumetricMoisture(-152.85, soil_2),
        volumetricMoisture(-152.85, soil_3),
    ]
    anaerobic = [
        volumetricMoisture(-0.1, soil_1),
        volumetricMoisture(-0.1, soil_2),
        volumetricMoisture(-0.1, soil_3),
    ]

    n_points = min(len(state_traj_mz1), len(state_traj_mz2), len(state_traj_mz3), len(crop_coeff), len(ref_evap))
    maximum_et = np.sum(crop_coeff[:n_points] * ref_evap[:n_points])
    if maximum_et <= 1e-12:
        return {"yield_mz1": np.nan, "yield_mz2": np.nan, "yield_mz3": np.nan, "yield_avg": np.nan}

    seasonal_et = np.zeros(3, dtype=float)
    trajectories = [state_traj_mz1, state_traj_mz2, state_traj_mz3]
    for idx in range(3):
        for i in range(n_points):
            ws = _water_stress_factor(
                trajectories[idx][i],
                pwp[idx],
                lower_mz[idx],
                upper_mz[idx],
                anaerobic[idx],
            )
            seasonal_et[idx] += ws * crop_coeff[i] * ref_evap[i]

    ym = params["yield_model"]
    yield_response_factor = float(ym["yield_response_factor"])
    maximal_potential_yield = float(ym["maximal_potential_yield"])
    yields = (1 - yield_response_factor + yield_response_factor * (seasonal_et / maximum_et)) * maximal_potential_yield

    return {
        "yield_mz1": float(yields[0]),
        "yield_mz2": float(yields[1]),
        "yield_mz3": float(yields[2]),
        "yield_avg": float(np.mean(yields)),
    }


def apply_corn_overrides(runner_module, params: Dict = CORN_PARAMS) -> None:
    runner_module.generate_Kc = lambda daily_avg_temp: generate_kc_maize(daily_avg_temp, params=params)
    runner_module.evaluate_yield_from_state_trajectories = (
        lambda s1, s2, s3: evaluate_yield_maize_from_state_trajectories(s1, s2, s3, params=params)
    )
