"""
Compute yield / irrigation / IWUE with paper-reproduction convention.

Overall convention (same as repository + paper-aligned reporting style):
    yield_overall = mean(Y1, Y2, Y3)                    [kg/m^2]
    irrigation_overall = I1 + I2 + I3                   [m^3/m^2]
    IWUE_overall = yield_overall / irrigation_overall   [kg/m^3]

No files are written. Results are printed to terminal only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class ZoneParams:
    lower_zone: float
    upper_zone: float
    pwp: float
    anaerobic: float


def _import_project_modules(script_dir: Path):
    common_dir = script_dir.parent / "common"
    actual_field_dir = common_dir / "actual_field"
    sys.path.insert(0, str(actual_field_dir))

    from crop_coefficient_calculation import generate_Kc  # type: ignore
    from parameters_mpc import (  # type: ignore
        ManagementZone_1,
        ManagementZone_2,
        ManagementZone_3,
    )
    from van_Genuchten_relations import volumetricMoisture  # type: ignore

    return {
        "common_dir": common_dir,
        "generate_Kc": generate_Kc,
        "ManagementZone_1": ManagementZone_1,
        "ManagementZone_2": ManagementZone_2,
        "ManagementZone_3": ManagementZone_3,
        "volumetricMoisture": volumetricMoisture,
    }


def evaluate_water_stress(theta_v: float, params: ZoneParams) -> float:
    if theta_v <= params.pwp:
        return 0.0
    if params.pwp < theta_v <= params.lower_zone:
        return (theta_v - params.pwp) / (params.lower_zone - params.pwp)
    if params.lower_zone < theta_v < params.upper_zone:
        return 1.0
    if params.upper_zone <= theta_v <= params.anaerobic:
        return (theta_v - params.upper_zone) / (params.anaerobic - params.upper_zone)
    return 0.0


def compute_zone_yield_kg_per_m2(
    state_traj: np.ndarray,
    params: ZoneParams,
    crop_coeff: np.ndarray,
    ref_evap: np.ndarray,
    y_max: float,
    ky: float,
) -> float:
    horizon = min(len(state_traj), len(crop_coeff), len(ref_evap))
    if horizon == 0:
        return float("nan")

    maximum_et = crop_coeff[:horizon] * ref_evap[:horizon]
    stress = np.array(
        [evaluate_water_stress(float(state_traj[t]), params) for t in range(horizon)]
    )
    seasonal_et = stress * maximum_et

    denom = float(np.sum(maximum_et))
    if denom <= 0.0:
        return float("nan")

    return float((1.0 - ky + ky * (float(np.sum(seasonal_et)) / denom)) * y_max)


def compute_zone_irrigation_m3_per_m2(input_mm_per_day: np.ndarray) -> float:
    # Repository saves irrigation as negative mm/day.
    return float(-np.sum(input_mm_per_day) / 1000.0)


def collect_zone_metrics(
    result_dir: Path,
    zone_params: Dict[int, ZoneParams],
    crop_coeff: np.ndarray,
    ref_evap: np.ndarray,
    y_max: float,
    ky: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for mz in (1, 2, 3):
        state_file = result_dir / f"StateCL_mz{mz}.txt"
        input_file = result_dir / f"InputCL_mz{mz}.txt"
        if not state_file.exists():
            raise FileNotFoundError(f"Missing file: {state_file}")
        if not input_file.exists():
            raise FileNotFoundError(f"Missing file: {input_file}")

        state = np.atleast_1d(np.loadtxt(state_file))
        irrigation_input = np.atleast_1d(np.loadtxt(input_file))

        y = compute_zone_yield_kg_per_m2(
            state_traj=state,
            params=zone_params[mz],
            crop_coeff=crop_coeff,
            ref_evap=ref_evap,
            y_max=y_max,
            ky=ky,
        )
        i = compute_zone_irrigation_m3_per_m2(irrigation_input)
        iwue = float(y / i) if i > 0 else float("nan")

        rows.append(
            {
                "scope": f"MZ{mz}",
                "yield_kg_per_m2": y,
                "irrigation_m3_per_m2": i,
                "iwue_kg_per_m3": iwue,
            }
        )
    return rows


def print_table(all_rows: List[Dict[str, float]]) -> None:
    header = (
        f"{'Approach':<16}"
        f"{'Scope':<12}"
        f"{'Yield (kg/m^2)':>18}"
        f"{'Irrigation (m^3/m^2)':>24}"
        f"{'IWUE (kg/m^3)':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in all_rows:
        print(
            f"{row['approach']:<16}"
            f"{row['scope']:<12}"
            f"{row['yield_kg_per_m2']:>18.6f}"
            f"{row['irrigation_m3_per_m2']:>24.6f}"
            f"{row['iwue_kg_per_m3']:>18.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute metrics with paper convention: overall IWUE = mean(yield) / sum(irrigation)."
        )
    )
    parser.add_argument(
        "--results",
        nargs="+",
        default=["results_cl-lbmpc", "results_cl-scmarl"],
        help="Result folder names under current directory.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=0.88,
        help="Maximum potential yield Ym in kg/m^2 for Appendix-C formula.",
    )
    parser.add_argument(
        "--ky",
        type=float,
        default=1.15,
        help="Yield response factor ky in Appendix-C formula.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    mods = _import_project_modules(script_dir)

    common_dir: Path = mods["common_dir"]
    generate_kc = mods["generate_Kc"]
    volumetric_moisture = mods["volumetricMoisture"]

    soil_1 = mods["ManagementZone_1"]()
    soil_2 = mods["ManagementZone_2"]()
    soil_3 = mods["ManagementZone_3"]()

    zone_params = {
        1: ZoneParams(
            lower_zone=0.200,
            upper_zone=0.281,
            pwp=float(volumetric_moisture(-152.85, soil_1)),
            anaerobic=float(volumetric_moisture(-0.1, soil_1)),
        ),
        2: ZoneParams(
            lower_zone=0.200,
            upper_zone=0.281,
            pwp=float(volumetric_moisture(-152.85, soil_2)),
            anaerobic=float(volumetric_moisture(-0.1, soil_2)),
        ),
        3: ZoneParams(
            lower_zone=0.230,
            upper_zone=0.301,
            pwp=float(volumetric_moisture(-152.85, soil_3)),
            anaerobic=float(volumetric_moisture(-0.1, soil_3)),
        ),
    }

    avg_temp = np.atleast_1d(np.loadtxt(common_dir / "weather_data" / "dAVG_TEMP.txt"))
    crop_coeff = np.atleast_1d(generate_kc(avg_temp))
    ref_evap = np.atleast_1d(np.loadtxt(common_dir / "weather_data" / "dPET.txt"))
    ref_evap = np.clip(ref_evap, 1.04, 9.0)

    all_rows: List[Dict[str, float]] = []
    for result_name in args.results:
        result_dir = (Path.cwd() / result_name).resolve()
        if not result_dir.exists():
            raise FileNotFoundError(f"Result folder not found: {result_dir}")

        zone_rows = collect_zone_metrics(
            result_dir=result_dir,
            zone_params=zone_params,
            crop_coeff=crop_coeff,
            ref_evap=ref_evap,
            y_max=args.ymax,
            ky=args.ky,
        )

        zone_yields = [r["yield_kg_per_m2"] for r in zone_rows]
        zone_irrig = [r["irrigation_m3_per_m2"] for r in zone_rows]

        field_yield = float(np.mean(zone_yields))
        field_irrig = float(np.sum(zone_irrig))
        field_iwue = float(field_yield / field_irrig) if field_irrig > 0 else float("nan")

        for r in zone_rows:
            all_rows.append({"approach": result_name, **r})
        all_rows.append(
            {
                "approach": result_name,
                "scope": "WholeField",
                "yield_kg_per_m2": field_yield,
                "irrigation_m3_per_m2": field_irrig,
                "iwue_kg_per_m3": field_iwue,
            }
        )

    print("Convention: IWUE_overall = mean(zone_yield) / sum(zone_irrigation)")
    print_table(all_rows)


if __name__ == "__main__":
    main()
