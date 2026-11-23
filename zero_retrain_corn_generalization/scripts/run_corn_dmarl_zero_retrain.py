import os
import sys
from datetime import datetime

from corn_overrides import CORN_PARAMS, apply_corn_overrides


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PERF_DIR = os.path.dirname(ROOT_DIR)
BASELINE_DIR = os.path.join(PERF_DIR, "scmarl_vs_dmarl")
CORE_DIR = os.path.join(BASELINE_DIR, "core_scmarl_vs_dmarl")

os.chdir(BASELINE_DIR)
if BASELINE_DIR not in sys.path:
    sys.path.append(BASELINE_DIR)
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

import season_long_runner_exp456 as runner
from scheduler_evaluation_dmarl import SolveScheduler


def main() -> None:
    apply_corn_overrides(runner, params=CORN_PARAMS)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rel_results_dir = os.path.join(
        "..",
        "zero_retrain_corn_generalization",
        "results",
        f"results_corn_dmarl_zero_retrain_{run_id}",
    )

    print("[INFO] Running zero-retrain corn experiment: DMARL")
    print(f"[INFO] results_dir={rel_results_dir}")
    metrics = runner.run_season_long_experiment(
        experiment_name="Zero-Retrain Corn Generalization - DMARL",
        solve_scheduler=SolveScheduler,
        results_dir_name=rel_results_dir,
        simulation_period=123,
    )
    print("[INFO] Finished.")
    print(metrics)


if __name__ == "__main__":
    main()

