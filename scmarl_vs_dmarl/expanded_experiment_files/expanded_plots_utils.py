import csv
import datetime as dt
import json
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, "core_scmarl_vs_dmarl")

# Keep legacy relative-path modules stable.
os.chdir(BASE_DIR)
sys.path.append(CORE_DIR)

from season_long_runner_exp456 import run_season_long_experiment  # noqa: E402


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _aggregate_group_metrics(group_metrics: List[Dict]) -> Dict:
    zone_yields: List[float] = []
    zone_irrig: List[float] = []
    for gm in group_metrics:
        zone_yields.extend([float(gm["yield_mz1"]), float(gm["yield_mz2"]), float(gm["yield_mz3"])])
        zone_irrig.extend([float(gm["irrigation_mz1"]), float(gm["irrigation_mz2"]), float(gm["irrigation_mz3"])])

    zone_yields_arr = np.asarray(zone_yields, dtype=float)
    zone_irrig_arr = np.asarray(zone_irrig, dtype=float)

    avg_yield = float(np.mean(zone_yields_arr))
    avg_irrig = float(np.mean(zone_irrig_arr))
    total_irrig = float(np.sum(zone_irrig_arr))
    total_yield = float(np.sum(zone_yields_arr))

    iwue_physical = float(total_yield / total_irrig) if total_irrig > 1e-12 else float("nan")
    iwue_paper_style = float(avg_yield / total_irrig) if total_irrig > 1e-12 else float("nan")

    per_zone_iwue = np.divide(
        zone_yields_arr,
        zone_irrig_arr,
        out=np.full_like(zone_yields_arr, np.nan),
        where=zone_irrig_arr > 1e-12,
    )

    return {
        "zone_count": int(len(zone_yields)),
        "avg_yield_kg_m2": avg_yield,
        "total_yield_sum_over_zones_kg_m2": total_yield,
        "avg_irrigation_m3_m2": avg_irrig,
        "total_irrigation_m3_m2": total_irrig,
        "iwue_physical_kg_m3": iwue_physical,
        "iwue_paper_style_kg_m3": iwue_paper_style,
        "per_zone_iwue_mean_kg_m3": float(np.nanmean(per_zone_iwue)),
        "per_zone_iwue_std_kg_m3": float(np.nanstd(per_zone_iwue)),
    }


def run_expanded_plots_experiment(
    method_name: str,
    solve_scheduler: Callable,
    zone_counts: List[int],
    setup_hook: Optional[Callable[[], None]] = None,
    simulation_period: int = 123,
) -> List[Dict]:
    run_id = _timestamp()
    out_root = os.path.join(BASE_DIR, f"results_{method_name.lower()}_expanded_{run_id}")
    _ensure_dir(out_root)

    all_rows: List[Dict] = []
    print(f"[INFO] Start expanded-plots experiment: method={method_name}, run_id={run_id}")
    print(f"[INFO] Output root: {out_root}")

    if setup_hook is not None:
        setup_hook()

    for zone_count in zone_counts:
        if zone_count % 3 != 0:
            raise ValueError(f"zone_count must be multiple of 3, got {zone_count}")

        groups = zone_count // 3
        print(f"\n[INFO] Running {method_name} for {zone_count} plots ({groups} groups x 3 plots)...")
        group_metrics: List[Dict] = []

        for g in range(groups):
            rel_results_dir = os.path.relpath(
                os.path.join(out_root, f"z{zone_count}_group{g + 1}"),
                BASE_DIR,
            )
            print(f"[INFO] -> Group {g + 1}/{groups}, results_dir={rel_results_dir}")
            metrics = run_season_long_experiment(
                experiment_name=f"{method_name} expanded z{zone_count} group{g + 1}",
                solve_scheduler=solve_scheduler,
                results_dir_name=rel_results_dir,
                simulation_period=simulation_period,
            )
            group_metrics.append(metrics)

        agg = _aggregate_group_metrics(group_metrics)
        agg["method"] = method_name
        agg["groups"] = groups
        agg["run_id"] = run_id
        all_rows.append(agg)

        print("[INFO] Aggregated:")
        print(
            f"  avg_yield={agg['avg_yield_kg_m2']:.4f} kg/m^2, "
            f"total_irrig={agg['total_irrigation_m3_m2']:.4f} m^3/m^2, "
            f"iwue_physical={agg['iwue_physical_kg_m3']:.4f} kg/m^3"
        )

    csv_path = os.path.join(out_root, f"summary_{method_name.lower()}_expanded.csv")
    json_path = os.path.join(out_root, f"summary_{method_name.lower()}_expanded.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "run_id",
                "zone_count",
                "groups",
                "avg_yield_kg_m2",
                "total_yield_sum_over_zones_kg_m2",
                "avg_irrigation_m3_m2",
                "total_irrigation_m3_m2",
                "iwue_physical_kg_m3",
                "iwue_paper_style_kg_m3",
                "per_zone_iwue_mean_kg_m3",
                "per_zone_iwue_std_kg_m3",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] CSV summary: {csv_path}")
    print(f"[INFO] JSON summary: {json_path}")
    return all_rows
