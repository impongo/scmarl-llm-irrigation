try:
    from .generate_schedules_exp_base import evaluate_scheduler_head_generic
except ImportError:
    from generate_schedules_exp_base import evaluate_scheduler_head_generic

_LLM_CALLBACK = None
_LLM_CALL_POLICY = "once_per_day"


def configure_llm(callback=None, call_policy="once_per_day"):
    global _LLM_CALLBACK, _LLM_CALL_POLICY
    _LLM_CALLBACK = callback
    _LLM_CALL_POLICY = call_policy


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
        coordinator_mode="llm_sync",
        local_agent_profile="scmarl_local",
        llm_callback=_LLM_CALLBACK,
        llm_call_policy=_LLM_CALL_POLICY,
    )
