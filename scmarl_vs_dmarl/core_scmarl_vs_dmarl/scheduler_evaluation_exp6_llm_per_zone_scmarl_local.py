try:
    from .generate_schedules_exp6_llm_per_zone_scmarl_local import configure_llm, evaluate_scheduler_head
except ImportError:
    from generate_schedules_exp6_llm_per_zone_scmarl_local import configure_llm, evaluate_scheduler_head

horizonLength = 14


def SolveScheduler(currentStates_mz1_all, currentStates_mz2_all, currentStates_mz3_all, refEvap_sequence, cropCoeff_sequence, lai_sequence, rain_sequence):
    stateArray_rz_mz1, stateArray_rz_mz2, stateArray_rz_mz3, irrigationDecisions, actionsArray_mz1, actionsArray_mz2, actionsArray_mz3 = evaluate_scheduler_head(
        currentStates_mz1_all,
        currentStates_mz2_all,
        currentStates_mz3_all,
        refEvap_sequence,
        cropCoeff_sequence,
        lai_sequence,
        rain_sequence,
        horizonLength,
    )
    print(irrigationDecisions)
    return actionsArray_mz1, actionsArray_mz2, actionsArray_mz3, irrigationDecisions, stateArray_rz_mz1, stateArray_rz_mz2, stateArray_rz_mz3
