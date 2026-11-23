import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(BASE_DIR, "core_scmarl_vs_dmarl")
os.chdir(BASE_DIR)
sys.path.append(CORE_DIR)

from expanded_plots_utils import run_expanded_plots_experiment  # noqa: E402
from scheduler_evaluation_scmarl import SolveScheduler  # noqa: E402


if __name__ == "__main__":
    run_expanded_plots_experiment(
        method_name="SCMARL",
        solve_scheduler=SolveScheduler,
        zone_counts=[9, 15],
        setup_hook=None,
        simulation_period=123,
    )

