"""Hard-coded if-else version of the corn coordinator.

Translates the THRESHOLD AIDS block of the corn EXP5 LLM prompt
(`run_corn_exp5_zero_retrain.py::build_prompt_exp5_corn`) into a deterministic
decision tree. Reuses the EXP5 scheduler entry point but injects a rule-based
callback instead of an LLM call.

Rule priority (matches the order in the corn EXP5 prompt's THRESHOLD AIDS):
    1. Strong open       : sensitivity in {HIGH,CRITICAL} & min_margin<0.006 & sdr_3d<0.80
    2. MID protection    : stage=MID & min_margin<0.002 & sdr_3d<0.95
    3. Weak open         : stage in {DEVELOPMENT,MID} & min_margin<0 & rain_tmr<etc_tmr
    4. Strong skip       : min_margin>=0.010 & rain_tmr>=etc_tmr
    5. Frequency damper  : days_since_last_open<=1 & min_margin>-0.003
    6. Conservative pull : HIGH/CRITICAL & min_margin<0.004 (lean to 1)
    7. Default           : 0
"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional

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
from scheduler_evaluation_exp5_llm_sync_scmarl_local import SolveScheduler, configure_llm


LLM_CALL_POLICY = "once_per_day"

STRONG_OPEN_MARGIN = 0.006
STRONG_OPEN_SDR = 0.80
MID_PROTECT_MARGIN = 0.002
MID_PROTECT_SDR = 0.95
STRONG_SKIP_MARGIN = 0.010
FREQ_DAMPER_MARGIN = -0.003
CONSERVATIVE_OPEN_MARGIN = 0.004

_SEASON_DAY_COUNTER = 0
_LAST_OPEN_DAY = -999
_OPEN_COUNT_7D = []
_OPEN_COUNT_14D = []


def _build_stage_from_day(day: int, stage_days: Dict[str, int]) -> str:
    d1 = stage_days["initial"]
    d2 = stage_days["development"]
    d3 = stage_days["mid"]
    if day <= d1:
        return "INITIAL"
    if day <= d1 + d2:
        return "DEVELOPMENT"
    if day <= d1 + d2 + d3:
        return "MID"
    return "LATE"


def _sensitivity_from_stage(stage: str) -> str:
    if stage == "MID":
        return "CRITICAL"
    if stage == "DEVELOPMENT":
        return "HIGH"
    if stage == "LATE":
        return "MEDIUM"
    return "LOW"


def _history_features(current_day: int) -> Dict[str, int]:
    global _OPEN_COUNT_7D, _OPEN_COUNT_14D
    _OPEN_COUNT_7D = [d for d in _OPEN_COUNT_7D if current_day - d < 7]
    _OPEN_COUNT_14D = [d for d in _OPEN_COUNT_14D if current_day - d < 14]
    days_since = 999 if _LAST_OPEN_DAY < 0 else (current_day - _LAST_OPEN_DAY)
    return {
        "days_since_last_open": int(days_since),
        "opens_last_7d": int(len(_OPEN_COUNT_7D)),
        "opens_last_14d": int(len(_OPEN_COUNT_14D)),
    }


def _corn_rule_decision(context: Dict, sim_day: int, history: Dict[str, int]) -> float:
    rz = [float(x) for x in context["root_zone_moisture"]]
    lb = [float(x) for x in context["lower_bounds"]]
    margin = [rz[i] - lb[i] for i in range(3)]
    min_margin = min(margin)

    stage = _build_stage_from_day(sim_day, CORN_PARAMS["stage_days"])
    sensitivity = _sensitivity_from_stage(stage)
    is_sensitive = sensitivity in ("HIGH", "CRITICAL")

    kc_today = float(context["current_inputs"]["crop_coeff"])
    pet_today = float(context["current_inputs"]["ref_et_mm_per_day"])
    etc_today = kc_today * pet_today

    fut_rain = [float(x) for x in context["future_inputs"]["rain_mm_per_day"]]
    fut_et = [float(x) for x in context["future_inputs"]["ref_et_mm_per_day"]]
    fut_kc = [float(x) for x in context["future_inputs"]["crop_coeff"]]
    n_fut = min(len(fut_rain), len(fut_et), len(fut_kc))

    n3 = min(3, n_fut)
    rain_3d = sum(fut_rain[:n3])
    etc_3d = sum(fut_et[i] * fut_kc[i] for i in range(n3))
    sdr_3d = rain_3d / etc_3d if etc_3d > 0.01 else 99.0

    if n_fut > 1:
        rain_tmr = fut_rain[1]
        etc_tmr = fut_et[1] * fut_kc[1]
    else:
        rain_tmr = 0.0
        etc_tmr = etc_today

    days_since_last_open = int(history.get("days_since_last_open", 999))

    if is_sensitive and min_margin < STRONG_OPEN_MARGIN and sdr_3d < STRONG_OPEN_SDR:
        return 1.0
    if stage == "MID" and min_margin < MID_PROTECT_MARGIN and sdr_3d < MID_PROTECT_SDR:
        return 1.0
    if stage in ("DEVELOPMENT", "MID") and min_margin < 0.0 and rain_tmr < etc_tmr:
        return 1.0
    if min_margin >= STRONG_SKIP_MARGIN and rain_tmr >= etc_tmr:
        return 0.0
    if days_since_last_open <= 1 and min_margin > FREQ_DAMPER_MARGIN:
        return 0.0
    if is_sensitive and min_margin < CONSERVATIVE_OPEN_MARGIN:
        return 1.0
    return 0.0


def rule_callback_corn(context: Dict) -> Optional[float]:
    global _SEASON_DAY_COUNTER, _LAST_OPEN_DAY, _OPEN_COUNT_7D, _OPEN_COUNT_14D
    _SEASON_DAY_COUNTER += 1
    sim_day = _SEASON_DAY_COUNTER

    history = _history_features(sim_day)
    decision = _corn_rule_decision(context, sim_day, history)

    if decision >= 0.5:
        _LAST_OPEN_DAY = sim_day
        _OPEN_COUNT_7D.append(sim_day)
        _OPEN_COUNT_14D.append(sim_day)
    return decision


def _reset_runtime_state() -> None:
    global _SEASON_DAY_COUNTER, _LAST_OPEN_DAY, _OPEN_COUNT_7D, _OPEN_COUNT_14D
    _SEASON_DAY_COUNTER = 0
    _LAST_OPEN_DAY = -999
    _OPEN_COUNT_7D = []
    _OPEN_COUNT_14D = []


def main() -> None:
    apply_corn_overrides(runner, params=CORN_PARAMS)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rel_results_dir = os.path.join(
        "..",
        "zero_retrain_corn_generalization",
        "results",
        f"results_corn_exp4_zero_retrain_{run_id}",
    )

    _reset_runtime_state()
    configure_llm(callback=rule_callback_corn, call_policy=LLM_CALL_POLICY)

    print("[INFO] Running zero-retrain corn experiment: EXP4 (hard-coded rule mirror of EXP5 prompt)")
    print(f"[INFO] results_dir={rel_results_dir}")
    metrics = runner.run_season_long_experiment(
        experiment_name="Zero-Retrain Corn Generalization - EXP4 Rule",
        solve_scheduler=SolveScheduler,
        results_dir_name=rel_results_dir,
        simulation_period=123,
    )
    print("[INFO] Finished.")
    print(metrics)


if __name__ == "__main__":
    main()
