try:
    from .generate_schedules_exp_base import evaluate_scheduler_head_generic
except ImportError:
    from generate_schedules_exp_base import evaluate_scheduler_head_generic

import numpy as np
from typing import Dict, Any


# Deterministic baseline from the paper: open only when today's rainfall is zero
# and at least one zone is below its lower soil-moisture bound.
def _exp4_rule_callback(context: Dict[str, Any]) -> float:
    margins = [
        float(moisture) - float(bound)
        for moisture, bound in zip(context["root_zone_moisture"], context["lower_bounds"])
    ]
    min_margin = min(margins)
    rain_today = float(context["current_inputs"]["rain_mm_per_day"])
    return float(rain_today == 0.0 and min_margin < 0.0)


def evaluate_scheduler_head(states_mz1, states_mz2, states_mz3, refEvap_sequence, cropCoeff_sequence, lai_sequence, rain_sequence, prediction_horizon):
    return evaluate_scheduler_head_generic(
        states_mz1=states_mz1,
        states_mz2=states_mz2,
        states_mz3=states_mz3,
        refEvap_sequence=refEvap_sequence,
        cropCoeff_sequence=cropCoeff_sequence,
        lai_sequence=lai_sequence,
        rain_sequence=rain_sequence,
        prediction_horizon=prediction_horizon,
        coordinator_mode="rule_sync",
        local_agent_profile="scmarl_local",
        llm_callback=_exp4_rule_callback,
        llm_call_policy="per_step",
    )
