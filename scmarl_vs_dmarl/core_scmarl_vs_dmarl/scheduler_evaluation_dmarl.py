
from generate_schedules_dmarl import evaluate_scheduler_head
from parameters_mpc import *
import numpy as np
import multiprocessing as mp 

#Load the  soil parameters for the MZs
soilPars_mz1 = ManagementZone_1()
soilPars_mz2 = ManagementZone_2()
soilPars_mz3 = ManagementZone_3()

horizonLength = 14 


def SolveScheduler(currentStates_mz1_all, currentStates_mz2_all, currentStates_mz3_all, refEvap_sequence, cropCoeff_sequence,lai_sequence, rain_sequence):
    

    # Determine the irrigation decision based on the semi-centralized multi-agent RL approach
    stateArray_rz_mz1,stateArray_rz_mz2,stateArray_rz_mz3,irrigationDecisions,actionsArray_mz1,actionsArray_mz2,actionsArray_mz3= \
            evaluate_scheduler_head(currentStates_mz1_all,currentStates_mz2_all,currentStates_mz3_all,refEvap_sequence,cropCoeff_sequence,lai_sequence, 
                            rain_sequence,horizonLength)

    print(irrigationDecisions)


    return actionsArray_mz1,actionsArray_mz2,actionsArray_mz3, irrigationDecisions, stateArray_rz_mz1,stateArray_rz_mz2,stateArray_rz_mz3
