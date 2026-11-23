import sys
sys.path.append("../common/actual_field")
from generate_discrete_decisions_lb_mpc import obtain_discrete_actions_mz1, obtain_discrete_actions_mz2, obtain_discrete_actions_mz3
from parameters_mpc import *
from formulate_scheduler import formulateOptimProb
import numpy as np
import os
import multiprocessing as mp 
from queue import Queue

#Load the  soil parameters for the MZs
soilPars_mz1 = ManagementZone_1()
soilPars_mz2 = ManagementZone_2()
soilPars_mz3 = ManagementZone_3()

#Load the scaling parameters
data_min_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
data_min_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz2_vrd_lai.txt')
data_max_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz2_vrd_lai.txt')
data_min_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz3_vrd_lai.txt')
data_max_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz3_vrd_lai.txt')
horizonLength = 14 


def _should_use_multiprocessing():
    # Windows uses "spawn" and each process rebuilds a very large CasADi graph.
    # Default to sequential execution on Windows to avoid memory spikes/std::bad_alloc.
    env_value = os.getenv("SCMARL_USE_MULTIPROCESS")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return os.name != "nt"


def _solve_optimizers_sequential(managementZone_indices, current_states, previousInputs, guesses_states, guesses_U,
                                 cropCoeff, refEvap, rooting_Depths, lai_factors, rain, disc_seq_lim):
    unsorted_result = []
    for i in range(3):
        local_qout = Queue()
        formulateOptimProb(
            i,
            managementZone_indices[i],
            current_states[i, :],
            previousInputs[i, :],
            guesses_states[i, :],
            guesses_U[i, :],
            cropCoeff,
            refEvap,
            rooting_Depths,
            lai_factors,
            rain,
            disc_seq_lim,
            local_qout,
        )
        unsorted_result.append(local_qout.get())
    return [t[1] for t in sorted(unsorted_result)]

def evaluate_limiting_MZ(irrig_decisions_mz1, irrig_decisions_mz2, irrig_decisions_mz3): #This will have to be improved
    if sum(irrig_decisions_mz1) == sum(irrig_decisions_mz2) == sum(irrig_decisions_mz3) == 0:
        return irrig_decisions_mz1
    else:
        for i in range(len(irrig_decisions_mz1)):
            if irrig_decisions_mz1[i] == 1.0:
                return irrig_decisions_mz1
        
            if irrig_decisions_mz2[i] == 1.0:
                return irrig_decisions_mz2
        
            if irrig_decisions_mz3[i] == 1.0:
                return irrig_decisions_mz3

def GetStateValues_mz1(optimalValues):
    predictedTrajectory = []
    stateSeedPoint = 15
    for j in range(horizonLength):
        predictedTrajectory.append(optimalValues[int(stateSeedPoint+j*(5))])
        pass
    predictedTrajectory_unscaled= (np.array(predictedTrajectory) * (data_max_mz1[0] - data_min_mz1[0])) + data_min_mz1[0]
    return predictedTrajectory_unscaled

def GetStateValues_mz2(optimalValues):
    predictedTrajectory = []
    stateSeedPoint = 15
    for j in range(horizonLength):
        predictedTrajectory.append(optimalValues[int(stateSeedPoint+j*(5))])
        pass
    predictedTrajectory_unscaled= (np.array(predictedTrajectory) * (data_max_mz2[0] - data_min_mz2[0])) + data_min_mz2[0]
    return predictedTrajectory_unscaled

def GetStateValues_mz3(optimalValues):
    predictedTrajectory = []
    stateSeedPoint = 15
    for j in range(horizonLength):
        predictedTrajectory.append(optimalValues[int(stateSeedPoint+j*(5))])
        pass
    predictedTrajectory_unscaled= (np.array(predictedTrajectory) * (data_max_mz3[0] - data_min_mz3[0])) + data_min_mz3[0]
    return predictedTrajectory_unscaled

def GetIrrigationAmount(optimalValues):
    prescribedIrrigation = []
    inputSeedPoint = 13
    for j in range(horizonLength):
        prescribedIrrigation.append(optimalValues[int(inputSeedPoint+j*(5))])
        pass
    return prescribedIrrigation

def SolveScheduler(currentStates_mz1_all, currentStates_mz2_all, currentStates_mz3_all, refEvap_sequence, cropCoeff_sequence,  rootingDepth_sequence, lai_sequence, rain_sequence, currentStates_mz1, currentStates_mz2,
                currentStates_mz3, previousInputs_mz1, previousInputs_mz2, previousInputs_mz3, cropCoeff, refEvap, rooting_Depths, lai_factors, rain):

    actions_disc_mz1, actions_cont_mz1, state_rz_mz1 = obtain_discrete_actions_mz1(currentStates_mz1_all, refEvap_sequence, cropCoeff_sequence,
                                                                                 rootingDepth_sequence, lai_sequence, rain_sequence,  horizonLength, soilPars_mz1) 
    actions_disc_mz2, actions_cont_mz2, state_rz_mz2 = obtain_discrete_actions_mz2(currentStates_mz2_all, refEvap_sequence, cropCoeff_sequence,
                                                                                 rootingDepth_sequence, lai_sequence, rain_sequence, horizonLength, soilPars_mz2)
    actions_disc_mz3, actions_cont_mz3, state_rz_mz3 = obtain_discrete_actions_mz3(currentStates_mz3_all, refEvap_sequence, cropCoeff_sequence, 
                                                                                rootingDepth_sequence, lai_sequence, rain_sequence, horizonLength, soilPars_mz3)  

    disc_seq_lim = evaluate_limiting_MZ(actions_disc_mz1, actions_disc_mz2, actions_disc_mz3) #Irrigation decision sequence of the limiting management zone
    state_rz_mz1 = (state_rz_mz1 - data_min_mz1[0])/(data_max_mz1[0] - data_min_mz1[0])
    state_rz_mz2 = (state_rz_mz2 - data_min_mz2[0])/(data_max_mz2[0] - data_min_mz2[0])
    state_rz_mz3 = (state_rz_mz3 - data_min_mz3[0])/(data_max_mz3[0] - data_min_mz3[0])

    current_states = np.zeros((3, len(currentStates_mz1)))
    current_states[0,:] = currentStates_mz1
    current_states[1,:] = currentStates_mz2
    current_states[2,:] = currentStates_mz3

    previousInputs = np.zeros((3, len(previousInputs_mz1)))
    previousInputs[0,:] = previousInputs_mz1
    previousInputs[1,:] = previousInputs_mz2
    previousInputs[2,:] = previousInputs_mz3

    guesses_states = np.zeros((3, len(state_rz_mz1[1:])))
    guesses_states[0,:] = state_rz_mz1[1:]
    guesses_states[1,:] = state_rz_mz2[1:]
    guesses_states[2,:] = state_rz_mz3[1:]

    guesses_U = np.zeros((3, len(actions_cont_mz1)))
    guesses_U[0,:] = actions_cont_mz1
    guesses_U[1,:] = actions_cont_mz2
    guesses_U[2,:] = actions_cont_mz3
    managementZone_indices = np.array([1,2,3]) # 3 management zones

    result = None
    if _should_use_multiprocessing():
        try:
            qout = mp.Queue()
            processes = [mp.Process(target=formulateOptimProb, args=(i, managementZone_indices[i], current_states[i,:], previousInputs[i,:],
                         guesses_states[i,:], guesses_U[i,:], cropCoeff, refEvap, rooting_Depths, lai_factors, rain, disc_seq_lim, qout)) for i in range(3)]
            
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            exit_codes = [p.exitcode for p in processes]
            if any(code != 0 for code in exit_codes):
                raise RuntimeError(f"Parallel optimizer process failed with exit codes: {exit_codes}")

            unsorted_result = [qout.get() for p in processes]
            result = [t[1] for t in sorted(unsorted_result)]
        except Exception as exc:
            print(f"[WARN] Parallel MPC solve failed ({exc}); fallback to sequential mode.")

    if result is None:
        result = _solve_optimizers_sequential(
            managementZone_indices,
            current_states,
            previousInputs,
            guesses_states,
            guesses_U,
            cropCoeff,
            refEvap,
            rooting_Depths,
            lai_factors,
            rain,
            disc_seq_lim,
        )
    
    #Obtain the irrigation rates, and the predicted states
    irrigationAmounts_mz1 = GetIrrigationAmount(result[0])
    states_mz1 =  GetStateValues_mz1(result[0])
    irrigationAmounts_mz2 = GetIrrigationAmount(result[1])
    states_mz2 =  GetStateValues_mz2(result[1])
    irrigationAmounts_mz3 = GetIrrigationAmount(result[2])
    states_mz3 =  GetStateValues_mz3(result[2])
    return irrigationAmounts_mz1, irrigationAmounts_mz2, irrigationAmounts_mz3, disc_seq_lim, states_mz1, states_mz2, states_mz3
