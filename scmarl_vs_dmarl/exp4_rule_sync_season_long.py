from core_scmarl_vs_dmarl.scheduler_evaluation_exp4_rule_sync_scmarl_local import SolveScheduler
from season_long_runner_exp456 import run_season_long_experiment

EXPERIMENT_NAME = "Experiment 4 - Rule Coordinator (Global 0/1) - SCMARL local agents"
RESULTS_DIR_NAME = "results_exp4_rule_sync_scmarl_local"


if __name__ == "__main__":
    run_season_long_experiment(
        experiment_name=EXPERIMENT_NAME,
        solve_scheduler=SolveScheduler,
        results_dir_name=RESULTS_DIR_NAME,
    )
