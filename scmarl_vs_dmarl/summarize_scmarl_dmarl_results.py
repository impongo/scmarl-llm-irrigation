import argparse
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
COMMON_ACTUAL_FIELD = (BASE_DIR / ".." / "common" / "actual_field").resolve()
WEATHER_DIR = (BASE_DIR / ".." / "common" / "weather_data").resolve()

sys.path.append(str(COMMON_ACTUAL_FIELD))

from crop_coefficient_calculation import generate_Kc  # noqa: E402
from parameters_mpc import ManagementZone_1, ManagementZone_2, ManagementZone_3  # noqa: E402
from van_Genuchten_relations import volumetricMoisture  # noqa: E402


MAX_POTENTIAL_YIELD = 8.8  # kg/m^2
YIELD_RESPONSE_FACTOR = 1.15


def _load_vector(path: Path) -> np.ndarray:
    arr = np.atleast_1d(np.loadtxt(path))
    return arr.astype(float).reshape(-1)


def _stress_factor(theta: float, lower: float, upper: float, pwp: float, anaerobic: float) -> float:
    if theta <= pwp:
        return 0.0
    if pwp < theta <= lower:
        return (theta - pwp) / (lower - pwp)
    if lower < theta < upper:
        return 1.0
    if upper <= theta <= anaerobic:
        return (theta - upper) / (anaerobic - upper)
    return 0.0


def _yield_from_state(
    state_traj: np.ndarray,
    crop_coeff: np.ndarray,
    ref_evap: np.ndarray,
    lower: float,
    upper: float,
    pwp: float,
    anaerobic: float,
) -> float:
    horizon = min(len(state_traj), len(crop_coeff), len(ref_evap))
    if horizon <= 0:
        return float("nan")

    state = state_traj[:horizon]
    cc = crop_coeff[:horizon]
    et0 = ref_evap[:horizon]

    ks = np.array([_stress_factor(v, lower, upper, pwp, anaerobic) for v in state])
    max_et = cc * et0
    seasonal_et = ks * max_et

    if np.sum(max_et) <= 0:
        return float("nan")

    return (1.0 - YIELD_RESPONSE_FACTOR + YIELD_RESPONSE_FACTOR * (np.sum(seasonal_et) / np.sum(max_et))) * MAX_POTENTIAL_YIELD


def _summarize_method(method_name: str, results_dir: Path, crop_coeff: np.ndarray, ref_evap: np.ndarray) -> None:
    state_mz1 = _load_vector(results_dir / "StateCL_mz1.txt")
    state_mz2 = _load_vector(results_dir / "StateCL_mz2.txt")
    state_mz3 = _load_vector(results_dir / "StateCL_mz3.txt")

    input_mz1 = _load_vector(results_dir / "InputCL_mz1.txt")
    input_mz2 = _load_vector(results_dir / "InputCL_mz2.txt")
    input_mz3 = _load_vector(results_dir / "InputCL_mz3.txt")

    soil_mz1 = ManagementZone_1()
    soil_mz2 = ManagementZone_2()
    soil_mz3 = ManagementZone_3()

    pwp_mz1 = volumetricMoisture(-152.85, soil_mz1)
    pwp_mz2 = volumetricMoisture(-152.85, soil_mz2)
    pwp_mz3 = volumetricMoisture(-152.85, soil_mz3)

    anaerobic_mz1 = volumetricMoisture(-0.1, soil_mz1)
    anaerobic_mz2 = volumetricMoisture(-0.1, soil_mz2)
    anaerobic_mz3 = volumetricMoisture(-0.1, soil_mz3)

    yield_mz1 = _yield_from_state(state_mz1, crop_coeff, ref_evap, 0.200, 0.281, pwp_mz1, anaerobic_mz1)
    yield_mz2 = _yield_from_state(state_mz2, crop_coeff, ref_evap, 0.200, 0.281, pwp_mz2, anaerobic_mz2)
    yield_mz3 = _yield_from_state(state_mz3, crop_coeff, ref_evap, 0.230, 0.301, pwp_mz3, anaerobic_mz3)
    yield_avg = float(np.nanmean([yield_mz1, yield_mz2, yield_mz3]))

    # InputCL is stored as mm/day with negative sign for irrigation. Convert to positive seasonal depth.
    irr_mm_mz1 = float(-np.sum(input_mz1))
    irr_mm_mz2 = float(-np.sum(input_mz2))
    irr_mm_mz3 = float(-np.sum(input_mz3))

    irr_m_mz1 = irr_mm_mz1 / 1000.0  # m^3/m^2
    irr_m_mz2 = irr_mm_mz2 / 1000.0
    irr_m_mz3 = irr_mm_mz3 / 1000.0

    irr_total_m = irr_m_mz1 + irr_m_mz2 + irr_m_mz3
    irr_avg_m = (irr_m_mz1 + irr_m_mz2 + irr_m_mz3) / 3.0

    iwue_paper = yield_avg / irr_total_m if irr_total_m > 0 else float("nan")
    iwue_physical = yield_avg / irr_avg_m if irr_avg_m > 0 else float("nan")

    irr_days_any = int(
        np.sum(
            (np.abs(input_mz1) > 1e-12)
            | (np.abs(input_mz2) > 1e-12)
            | (np.abs(input_mz3) > 1e-12)
        )
    )

    print(f"\n=== {method_name} ===")
    print(f"Results folder: {results_dir}")
    print(f"Irrigation days (any MZ): {irr_days_any}")
    print(f"Yield MZ1/MZ2/MZ3 (kg/m^2): {yield_mz1:.4f}, {yield_mz2:.4f}, {yield_mz3:.4f}")
    print(f"Average yield (kg/m^2): {yield_avg:.4f}")
    print(f"Irrigation MZ1/MZ2/MZ3 (m^3/m^2): {irr_m_mz1:.4f}, {irr_m_mz2:.4f}, {irr_m_mz3:.4f}")
    print(f"Total irrigation (sum over 3 MZs, m^3/m^2): {irr_total_m:.4f}")
    print(f"Average irrigation (across 3 MZs, m^3/m^2): {irr_avg_m:.4f}")
    print(f"IWUE (paper convention = avg_yield / total_irrigation, kg/m^3): {iwue_paper:.4f}")
    print(f"IWUE (physical convention = avg_yield / avg_irrigation, kg/m^3): {iwue_physical:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DMARL/SCMARL seasonal results.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dmarl", "scmarl"],
        help="Methods to summarize. Supported: dmarl, scmarl",
    )
    args = parser.parse_args()

    avg_temp = _load_vector(WEATHER_DIR / "dAVG_TEMP.txt")
    ref_evap = _load_vector(WEATHER_DIR / "dPET.txt")
    ref_evap = np.clip(ref_evap, 1.04, 9.0)
    crop_coeff = generate_Kc(avg_temp)

    for method in args.methods:
        method = method.lower().strip()
        if method not in {"dmarl", "scmarl"}:
            print(f"[WARN] Skip unsupported method: {method}")
            continue
        res_dir = BASE_DIR / f"results_{method}"
        if not res_dir.exists():
            print(f"[WARN] Results folder not found for {method}: {res_dir}")
            continue
        try:
            _summarize_method(method.upper(), res_dir, crop_coeff, ref_evap)
        except Exception as exc:
            print(f"[ERROR] Failed to summarize {method}: {exc}")


if __name__ == "__main__":
    main()
