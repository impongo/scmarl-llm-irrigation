import csv
import glob
import os
from datetime import datetime

import numpy as np

from corn_overrides import evaluate_yield_maize_from_state_trajectories


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
OUT_DIR = os.path.join(ROOT_DIR, "results")


def _latest_result_dir(method: str) -> str:
    pattern = os.path.join(RESULTS_DIR, f"results_corn_{method}_zero_retrain_*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No result directories found for method={method}")
    return candidates[-1]


def _load_vec(path: str) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=float)).reshape(-1)


def summarize_one(method: str) -> dict:
    res_dir = _latest_result_dir(method)
    s1 = _load_vec(os.path.join(res_dir, "StateCL_mz1.txt"))
    s2 = _load_vec(os.path.join(res_dir, "StateCL_mz2.txt"))
    s3 = _load_vec(os.path.join(res_dir, "StateCL_mz3.txt"))
    i1 = _load_vec(os.path.join(res_dir, "InputCL_mz1.txt"))
    i2 = _load_vec(os.path.join(res_dir, "InputCL_mz2.txt"))
    i3 = _load_vec(os.path.join(res_dir, "InputCL_mz3.txt"))

    y = evaluate_yield_maize_from_state_trajectories(s1, s2, s3)
    irr1 = -float(np.sum(i1) / 1000.0)
    irr2 = -float(np.sum(i2) / 1000.0)
    irr3 = -float(np.sum(i3) / 1000.0)
    irr_avg = (irr1 + irr2 + irr3) / 3.0
    iwue = y["yield_avg"] / irr_avg if irr_avg > 1e-12 else np.nan

    return {
        "method": method,
        "results_dir": res_dir,
        "yield_mz1": y["yield_mz1"],
        "yield_mz2": y["yield_mz2"],
        "yield_mz3": y["yield_mz3"],
        "yield_avg_kg_m2": y["yield_avg"],
        "irrig_mz1_m3_m2": irr1,
        "irrig_mz2_m3_m2": irr2,
        "irrig_mz3_m3_m2": irr3,
        "irrig_avg_m3_m2": irr_avg,
        "iwue_kg_m3": iwue,
    }


def main() -> None:
    rows = [summarize_one("scmarl"), summarize_one("dmarl"), summarize_one("exp5")]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"corn_zero_retrain_summary_{ts}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Summary CSV: {out_csv}")
    for r in rows:
        print(
            f"{r['method']}: yield_avg={r['yield_avg_kg_m2']:.4f} kg/m^2, "
            f"irrig_avg={r['irrig_avg_m3_m2']:.4f} m^3/m^2, iwue={r['iwue_kg_m3']:.4f} kg/m^3"
        )


if __name__ == "__main__":
    main()

