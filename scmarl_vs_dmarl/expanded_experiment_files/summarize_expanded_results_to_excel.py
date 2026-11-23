import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(BASE_DIR, "..", "common")
ACTUAL_FIELD_DIR = os.path.join(COMMON_DIR, "actual_field")
sys.path.append(ACTUAL_FIELD_DIR)

from crop_coefficient_calculation import generate_Kc
from temporal_spatial_parameters_field import pars_mz1, pars_mz2, pars_mz3
from van_Genuchten_relations import volumetricMoisture

METHOD_DIRS = {
    "scmarl": os.path.join(BASE_DIR, "results_scmarl_expanded"),
    "dmarl": os.path.join(BASE_DIR, "results_dmarl_expanded"),
    "exp5": os.path.join(BASE_DIR, "results_exp5_expanded"),
}

GROUP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SOIL_PARS = [pars_mz1(), pars_mz2(), pars_mz3()]
LOWER_MZ = [0.200, 0.200, 0.230]
UPPER_MZ = [0.281, 0.281, 0.301]
PWP = [volumetricMoisture(-152.85, SOIL_PARS[0]), volumetricMoisture(-152.85, SOIL_PARS[1]), volumetricMoisture(-152.85, SOIL_PARS[2])]
ANAEROBIC = [volumetricMoisture(-0.1, SOIL_PARS[0]), volumetricMoisture(-0.1, SOIL_PARS[1]), volumetricMoisture(-0.1, SOIL_PARS[2])]
MAXIMAL_POTENTIAL_YIELD = 8.8
YIELD_RESPONSE_FACTOR = 1.15


def _water_stress_factor(current_vol_moisture: float, pwp: float, lower_zone: float, upper_zone: float, anaerobic_point: float) -> float:
    if current_vol_moisture <= pwp:
        return 0.0
    if pwp < current_vol_moisture <= lower_zone:
        return (current_vol_moisture - pwp) / (lower_zone - pwp)
    if lower_zone < current_vol_moisture < upper_zone:
        return 1.0
    if upper_zone <= current_vol_moisture <= anaerobic_point:
        return (current_vol_moisture - upper_zone) / (anaerobic_point - upper_zone)
    return 0.0


def _evaluate_yields_from_states(state_traj_mz1: np.ndarray, state_traj_mz2: np.ndarray, state_traj_mz3: np.ndarray) -> Dict[str, float]:
    weather_avg_temp = np.loadtxt(os.path.join(COMMON_DIR, "weather_data", "dAVG_TEMP.txt"))
    crop_coeff = generate_Kc(weather_avg_temp)
    ref_evap = np.loadtxt(os.path.join(COMMON_DIR, "weather_data", "dPET.txt"))
    ref_evap = np.clip(ref_evap, 1.04, 9.0)

    n_points = min(len(state_traj_mz1), len(state_traj_mz2), len(state_traj_mz3), len(crop_coeff), len(ref_evap))
    maximum_et = np.sum(crop_coeff[:n_points] * ref_evap[:n_points])

    seasonal_et = np.zeros(3)
    trajectories = [state_traj_mz1, state_traj_mz2, state_traj_mz3]
    for idx in range(3):
        for i in range(n_points):
            ws = _water_stress_factor(trajectories[idx][i], PWP[idx], LOWER_MZ[idx], UPPER_MZ[idx], ANAEROBIC[idx])
            seasonal_et[idx] += ws * crop_coeff[i] * ref_evap[i]

    yields = (1 - YIELD_RESPONSE_FACTOR + YIELD_RESPONSE_FACTOR * (seasonal_et / maximum_et)) * MAXIMAL_POTENTIAL_YIELD
    return {
        "yield_mz1": float(yields[0]),
        "yield_mz2": float(yields[1]),
        "yield_mz3": float(yields[2]),
    }


def _parse_group_name(folder_name: str) -> Tuple[int, int]:
    m = re.match(r"z(\d+)_group(\d+)$", folder_name)
    if m is None:
        raise ValueError(f"Invalid group folder name: {folder_name}")
    return int(m.group(1)), int(m.group(2))


def _load_zone_metrics(group_dir: str) -> List[Dict]:
    state_mz1 = np.loadtxt(os.path.join(group_dir, "StateCL_mz1.txt"), dtype=float)
    state_mz2 = np.loadtxt(os.path.join(group_dir, "StateCL_mz2.txt"), dtype=float)
    state_mz3 = np.loadtxt(os.path.join(group_dir, "StateCL_mz3.txt"), dtype=float)
    input_mz1 = np.loadtxt(os.path.join(group_dir, "InputCL_mz1.txt"), dtype=float)
    input_mz2 = np.loadtxt(os.path.join(group_dir, "InputCL_mz2.txt"), dtype=float)
    input_mz3 = np.loadtxt(os.path.join(group_dir, "InputCL_mz3.txt"), dtype=float)

    yield_metrics = _evaluate_yields_from_states(
        np.asarray(state_mz1, dtype=float),
        np.asarray(state_mz2, dtype=float),
        np.asarray(state_mz3, dtype=float),
    )

    irrig_mz1 = -float(np.sum(np.asarray(input_mz1, dtype=float)) / 1000.0)
    irrig_mz2 = -float(np.sum(np.asarray(input_mz2, dtype=float)) / 1000.0)
    irrig_mz3 = -float(np.sum(np.asarray(input_mz3, dtype=float)) / 1000.0)

    rows = []
    for idx, (yield_v, irrig_v) in enumerate(
        [
            (yield_metrics["yield_mz1"], irrig_mz1),
            (yield_metrics["yield_mz2"], irrig_mz2),
            (yield_metrics["yield_mz3"], irrig_mz3),
        ],
        start=1,
    ):
        iwue = float(yield_v / irrig_v) if irrig_v > 1e-12 else np.nan
        rows.append(
            {
                "zone_index": idx,
                "yield_kg_m2": float(yield_v),
                "irrigation_m3_m2": float(irrig_v),
                "iwue_kg_m3": iwue,
            }
        )
    return rows


def build_rows_for_zone_count(zone_count: int) -> List[Dict]:
    out_rows: List[Dict] = []
    methods_order = ["scmarl", "dmarl", "exp5"]

    for method in methods_order:
        method_dir = METHOD_DIRS[method]
        group_dirs = []
        for name in os.listdir(method_dir):
            full_path = os.path.join(method_dir, name)
            if not os.path.isdir(full_path):
                continue
            zc, gi = _parse_group_name(name)
            if zc == zone_count:
                group_dirs.append((gi, full_path))
        group_dirs.sort(key=lambda x: x[0])

        if not group_dirs:
            raise FileNotFoundError(f"No data found for method={method}, zone_count={zone_count}")

        method_yields = []
        method_irrigs = []

        for group_idx, group_dir in group_dirs:
            zone_metrics = _load_zone_metrics(group_dir)
            group_char = GROUP_LABELS[group_idx - 1]
            for zone_metric in zone_metrics:
                plot_name = f"{group_char}组：{zone_metric['zone_index']}号地"
                row = {
                    "方法": method,
                    "地块": plot_name,
                    "产量kg/m²": zone_metric["yield_kg_m2"],
                    "总灌溉量m³/m²": zone_metric["irrigation_m3_m2"],
                    "IWUE kg/m³": zone_metric["iwue_kg_m3"],
                }
                out_rows.append(row)
                method_yields.append(zone_metric["yield_kg_m2"])
                method_irrigs.append(zone_metric["irrigation_m3_m2"])

        total_yield = float(np.sum(method_yields))
        total_irrig = float(np.sum(method_irrigs))
        iwue_total = float(total_yield / total_irrig) if total_irrig > 1e-12 else np.nan
        out_rows.append(
            {
                "方法": method,
                "地块": "整块地汇总",
                "产量kg/m²": total_yield,
                "总灌溉量m³/m²": total_irrig,
                "IWUE kg/m³": iwue_total,
            }
        )

    return out_rows


def main() -> None:
    rows_9 = build_rows_for_zone_count(9)
    rows_15 = build_rows_for_zone_count(15)

    df_9 = pd.DataFrame(rows_9)
    df_15 = pd.DataFrame(rows_15)

    output_path = os.path.join(BASE_DIR, "expanded_results_summary.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_9.to_excel(writer, index=False, sheet_name="9块地")
        df_15.to_excel(writer, index=False, sheet_name="15块地")

    print(output_path)


if __name__ == "__main__":
    main()
