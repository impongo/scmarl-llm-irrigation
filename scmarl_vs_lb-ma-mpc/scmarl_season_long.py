import sys
import os


sys.path.append("core_scmarl_vs_lb-mpc")
sys.path.append("../common/estimator_design")
sys.path.append("trained_lstm_models")
sys.path.append("../common/actual_field")

import numpy as np
from matplotlib import pyplot as plt
from scheduler_formulation_scmarl import SolveScheduler
from lstm_mz1 import getRNNmodel_numpy_mz1
from lstm_mz2 import getRNNmodel_numpy_mz2
from lstm_mz3 import getRNNmodel_numpy_mz3
from richards_equation_field import volMoistureAllNodes
from temporal_spatial_parameters_field import *
from model_simulation import simulate_Richards_equation
from crop_coefficient_calculation import generate_Kc
from van_Genuchten_relations import pressureHead
from ekf_algorithm import discreteTimeEKF
from richards_equation_ekf import rz_moisture_alg,surface_moisture_alg
from yield_evaluation_scmarl_scheduler import evaluate_yield
totalDepth, axialNodes, totalNodes= spatialVariables_1D_act()
samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps = temporalVariable()
soilPars_mz1 = pars_mz1()
soilPars_mz2 = pars_mz2()
soilPars_mz3 = pars_mz3()
sequenceLength = 5

#create a folder to store the simulation results
path_results_cl = "results_cl-scmarl"


isExist_results_cl = os.path.exists(path_results_cl)

if not isExist_results_cl:
    os.makedirs(path_results_cl)


# Initialize the Kalman Filter
measurement_covariance = 19.25
process_covariance = 0.05*np.eye(21)
# covariance = 1.90*np.eye(21)
covariance_mz1 = 15.9*np.ones((21,21))
covariance_mz2 = 15.9*np.ones((21,21))
covariance_mz3 = 15.9*np.ones((21,21))

#Some user defined functions that introduce uncertainty in the weather conditions
def evaluate_weather_uncertainy_ET(ref_Evap_sequence, prediction_horizon):
    noise = []
    for i in range(prediction_horizon):
        _std_dev = (i+1)*0.002*1e-09 # previous factor 0.02
        np.random.seed(i)
        _noise = np.random.normal(0, _std_dev)
        noise.append(_noise)
        pass 
    ref_Evap_sequence_noisy = ref_Evap_sequence*(86400*1000) + np.array(noise)*(86400*1000)
    for i in range(len(ref_Evap_sequence_noisy)):
        if ref_Evap_sequence_noisy[i]< 1.04:
            ref_Evap_sequence_noisy[i] = 1.04
        if ref_Evap_sequence_noisy[i] > 9.0:
            ref_Evap_sequence_noisy[i] = 9.0
    pass
    return ref_Evap_sequence_noisy/(86400*1000)

def evaluate_weather_uncertainty_rain(rain_sequence, prediction_horizon):
    noise = []
    for i in range(prediction_horizon):
        _std_dev = (i+1)/21 #previous value,  14.0, 7.0
        np.random.seed(i)
        _noise = np.random.normal(0, _std_dev)
        noise.append(_noise)
        pass 
    rain_sequence_noisy = np.zeros(len(rain_sequence))
    noise = np.array(noise)

    for i in range(prediction_horizon):
        if rain_sequence[i] == 0:
            rain_sequence_noisy[i]=0 
        else:
            rain_sequence_noisy[i] = -1*rain_sequence[i]*1000 + noise[i]
    rain_sequence_noisy = np.abs(rain_sequence_noisy)
    rain_sequence_noisy[rain_sequence_noisy<0.5] = 0.0
    return -rain_sequence_noisy/1000


def main():
    global covariance_mz1, covariance_mz2, covariance_mz3
    #Load the weather conditions
    # _average_temps = _weather_conditions[:,0]
    # _crop_coeff = generate_Kc(_average_temps)
    # _rain_act = _weather_conditions[:,1]
    # _ref_Evap = _weather_conditions[:,2]
    # # #Add noise to the rain
    _average_temps = np.loadtxt('../common/weather_data/dAVG_TEMP.txt')
    _crop_coeff = generate_Kc(_average_temps)
    # _rain_act = _weather_conditions[:,1]
    # _ref_Evap = _weather_conditions[:,2]
    _rain_act = np.loadtxt('../common/weather_data/dRAIN.txt')
    _ref_Evap = np.loadtxt('../common/weather_data/dPET.txt')
    # #Add noise to the rain
    # np.random.seed(5)
    # noise_rain_manual = []
    # for i in range(len(_rain_act)):
    #     if _rain_act[i]!=0:
    #         noise_rain_manual.append(np.random.normal(0,2.0))
    #     else:
    #         noise_rain_manual.append(np.random.normal(0, 0.05))
    #         pass
    #     pass
    # rain_noisy = _rain_act + np.array(noise_rain_manual)
    # rain_noisy = np.abs(rain_noisy)
    # rain_noisy[rain_noisy<0.5] = 0.0
    
    # _rain_predicted = rain_noisy
    _rain_predicted = _rain_act
    _ref_Evap_predicted = _ref_Evap
    
    # #Add noise to the reference evapotranspiration
    # np.random.seed(10)
    # noise = np.random.normal(0, 8e-09, len(_ref_Evap))
    # noise = noise*(86400*1000)
    # _ref_Evap_predicted = _ref_Evap + noise
    
    # for i in range(sequenceLength-1):
    #     _ref_Evap_predicted[i] =_ref_Evap[i]
    #     pass
    
    for i in range(len(_ref_Evap_predicted)):
        if _ref_Evap_predicted[i]< 1.04:
            _ref_Evap_predicted[i] = 1.04
        if _ref_Evap_predicted[i] > 9.0:
            _ref_Evap_predicted[i] = 9.0
        pass
    
    for i in range(len(_ref_Evap)):
        if _ref_Evap[i]< 1.04:
            _ref_Evap[i] = 1.04
        if _ref_Evap[i] > 9.0:
            _ref_Evap[i] = 9.0
        pass
    
    def compute_root_zone_moisture(Xk, rooting_depth):
        if rooting_depth == 1.00: 
            root_zone_head = 0.10*((1/6)*(Xk[0] + Xk[1] + Xk[2] + Xk[3] + Xk[4] + Xk[5]))  + \
                0.20*((1/6)*(Xk[5] + Xk[6] + Xk[7] + Xk[8] + Xk[9] + Xk[10])) + \
                0.30*((1/11)*(Xk[10] + Xk[11] + Xk[12] + Xk[13] + Xk[14] + Xk[15] + Xk[16] + Xk[17] + Xk[18] + Xk[19] + Xk[20])) + \
                0.40*((1/11)*(Xk[20] + Xk[21] + Xk[22] + Xk[23] + Xk[24] + Xk[25] + Xk[26] + Xk[27] + Xk[28] + Xk[29] + Xk[30]))
        else:    
            root_zone_head = 0.10*((1/6)*(Xk[10] + Xk[11] + Xk[12] + Xk[13] + Xk[14] + Xk[15])) +\
                0.20*((1/6)*(Xk[15] + Xk[16] + Xk[17] + Xk[18] + Xk[19] + Xk[20])) +\
                0.30*((1/6)*(Xk[20] + Xk[21] + Xk[22] + Xk[23] + Xk[24] + Xk[25])) + \
                0.40*((1/6)*(Xk[25] + Xk[26] + Xk[27] + Xk[28] + Xk[29] + Xk[30]))
        return root_zone_head   
    
    #Converts the root zone pressure head to soil moisture
    def obtain_soil_moisture(head, soilPars):
        vol_moist = volMoistureAllNodes(head, soilPars)
        return vol_moist
    
    #Load the scaling data for the simulation
    data_min_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
    data_max_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
    data_min_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz2_vrd_lai.txt')
    data_max_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz2_vrd_lai.txt')
    data_min_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz3_vrd_lai.txt')
    data_max_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz3_vrd_lai.txt')
    
    #Load the initial states and uncontrolled inputs (i.e. ET and Crop Coeff)
    initialState_mz1 = np.loadtxt("../common/initial_states/Init_state_rz_mz1_vrd_lai.txt")
    initialState_mz2 = np.loadtxt("../common/initial_states/Init_state_rz_mz2_vrd_lai.txt")
    initialState_mz3 = np.loadtxt("../common/initial_states/Init_state_rz_mz3_vrd_lai.txt")
    
    initialIrrigation_mz1 = np.loadtxt("../common/initial_states/Init_irrig_mz1_vrd_lai.txt")
    initialIrrigation_mz2 = np.loadtxt("../common/initial_states/Init_irrig_mz2_vrd_lai.txt")
    initialIrrigation_mz3 = np.loadtxt("../common/initial_states/Init_irrig_mz3_vrd_lai.txt")
    
    initialIrrigation_mz1 = initialIrrigation_mz1*86400
    initialIrrigation_mz2 = initialIrrigation_mz2*86400
    initialIrrigation_mz3 = initialIrrigation_mz3*86400
    
    
    
    referenceEvap_all_mz1 = _ref_Evap_predicted/(86400*1000)
    referenceEvap_all_act_mz1 = _ref_Evap/(86400*1000)
    cropCoefficient_all_mz1 = _crop_coeff
    rain_all_predicted = -1*(_rain_predicted/1000)
    rain_all_act = -1*(_rain_act/1000)
    # lai_factors_all = np.loadtxt("./relevant_data/LAI_factors.txt")
    lai_factors_all = np.ones(180)
    
    #Pressure head values for the entire soil column, for the 3 MZs
    allHeadValues_mz1 = np.loadtxt("../common/initial_states/Init_state_all_mz1_vrd_lai.txt")
    allHeadValues_mz2 = np.loadtxt("../common/initial_states/Init_state_all_mz2_vrd_lai.txt")
    allHeadValues_mz3 = np.loadtxt("../common/initial_states/Init_state_all_mz3_vrd_lai.txt")
    
    
    #Relevant for the simulation
    sequenceLength = 5 
    horizonLength_orig = 14
    horizonLength_weather = int(sequenceLength + horizonLength_orig)
    
    simulationPeriod = 123 #Can be tuned
    rooting_depths = np.zeros(160)
    rooting_depths[0:72] = 0.50 
    rooting_depths[72:] = 0.50
    
    #Lists to store the simulation results
    statesList_mz1 = []
    statesList_mz2 = []
    statesList_mz3 = []
    irrigationList_mz1 = []
    irrigationList_mz2 = []
    irrigationList_mz3 = []
    
    for j in range(sequenceLength-1):
        statesList_mz1+=[initialState_mz1[j]]
        statesList_mz2+=[initialState_mz2[j]]
        statesList_mz3+=[initialState_mz3[j]]
        pass
    
    j=4
    
    statesList_mz1+=[compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1),0.50)]
    statesList_mz2+=[compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz2,soilPars_mz2),0.50)]
    statesList_mz3+=[compute_root_zone_moisture(obtain_soil_moisture(1.08*allHeadValues_mz3,soilPars_mz3),0.50)]
    
    factor_mz1=compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1),0.50)/initialState_mz1[4]
    factor_mz2=compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz2,soilPars_mz2),0.50)/initialState_mz2[4]
    factor_mz3=compute_root_zone_moisture(obtain_soil_moisture(1.08*allHeadValues_mz3,soilPars_mz3),0.50)/initialState_mz3[4]
    
    for j in range(sequenceLength-1):
        irrigationList_mz1+=[initialIrrigation_mz1[j]]
        irrigationList_mz2+=[initialIrrigation_mz2[j]]
        irrigationList_mz3+=[initialIrrigation_mz3[j]]
        pass
     
    #Lists to store the simulation results
    prescribedIrrigation_closedLoop_mz1 = []
    prescribedIrrigation_closedLoop_mz2 = []
    prescribedIrrigation_closedLoop_mz3 = []
    
    stateTrajectory_closedLoop_mz1 = []
    stateTrajectory_closedLoop_mz2 = []
    stateTrajectory_closedLoop_mz3 = []
    
    stateTrajectory_openLoop_mz1 = []
    stateTrajectory_openLoop_mz2 = []
    stateTrajectory_openLoop_mz3 = []
    
    irrigationDecisions_closedLoop = []
    
    stateTrajectory_closedLoop_mz1.append(initialState_mz1[-1])
    stateTrajectory_closedLoop_mz2.append(initialState_mz2[-1])
    stateTrajectory_closedLoop_mz3.append(initialState_mz3[-1])
    
    stateTrajectory_openLoop_mz1.append(compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1),0.50))
    stateTrajectory_openLoop_mz2.append(compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz2,soilPars_mz2),0.50))
    stateTrajectory_openLoop_mz3.append(compute_root_zone_moisture(obtain_soil_moisture(1.08*allHeadValues_mz3,soilPars_mz3),0.50))
    
    
    #For the simulation of the actual plant, the Richards Equation in this work
    allStates_mz1=np.zeros((simulationPeriod+1, totalNodes))
    allStates_mz2=np.zeros((simulationPeriod+1, totalNodes))
    allStates_mz3=np.zeros((simulationPeriod+1, totalNodes))
    
    allStates_mz1_est=np.zeros((simulationPeriod+1, totalNodes))
    allStates_mz2_est=np.zeros((simulationPeriod+1, totalNodes))
    allStates_mz3_est=np.zeros((simulationPeriod+1, totalNodes))
    
    allStates_mz1[0,:] = allHeadValues_mz1
    allStates_mz2[0,:] = allHeadValues_mz2 
    allStates_mz3[0,:] = allHeadValues_mz3
    
    allStates_mz1_est[0,:] = 1.05*allHeadValues_mz1
    allStates_mz2_est[0,:] = 1.05*allHeadValues_mz2 
    allStates_mz3_est[0,:] = 1.08*allHeadValues_mz3
    
    #Lower and upper bounds of the target zone for each MZ
    LowerZone_mz1 = 0.200 
    UpperZone_mz1 = 0.280 #0.280 
    LowerZone_mz2 = 0.200 
    UpperZone_mz2 = 0.280 #0.280 
    LowerZone_mz3 = 0.230 
    UpperZone_mz3 = 0.300 #0.300
    
    rain_noisy_all = np.zeros((simulationPeriod, horizonLength_weather))
    refEvap_noisy_all = np.zeros((simulationPeriod, horizonLength_weather))
    
    
    for k in range(simulationPeriod):
        print("\n\n")
        print("------------Open-loop simulation for day", k+1)
        # Use a rolling day index within each local weather window.
        current_day_offset = sequenceLength - 1
        horizon_slice = slice(current_day_offset, current_day_offset + horizonLength_orig)

        x0_init_mz1 = np.array(statesList_mz1[k : k+sequenceLength])
        u_init_mz1  = np.array(irrigationList_mz1[k : k+sequenceLength-1])
        kc_init_mz1 = cropCoefficient_all_mz1[k : k+horizonLength_weather]
        et_init_mz1 = referenceEvap_all_mz1[k : k+horizonLength_weather]
        et_init_mz1_noisy = evaluate_weather_uncertainy_ET(et_init_mz1, len(et_init_mz1))
        # et_init_mz1_noisy = et_init_mz1
        et_init_mz1_act = referenceEvap_all_act_mz1[k : k+horizonLength_weather]
        rain = rain_all_predicted[k : k+horizonLength_weather]
        rain_act = rain_all_act[k : k+horizonLength_weather]
        rooting_depth = rooting_depths[k:k+horizonLength_weather]
        rain_noisy = evaluate_weather_uncertainty_rain(rain, len(rain))
        # rain_noisy = rain
        rain_noisy_all[k,:] = rain_noisy
        refEvap_noisy_all[k,:] = et_init_mz1_noisy
        lai_factors = lai_factors_all[k:k+horizonLength_weather]
    
        x0_scaled_mz1 = (x0_init_mz1 - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
        kc_scaled_mz1 = (kc_init_mz1 - data_min_mz1[2])/(data_max_mz1[2]-data_min_mz1[2])
        #et_scaled_mz1 = (et_init_mz1 - data_min_mz1[3])/(data_max_mz1[3]-data_min_mz1[3])
        et_scaled_mz1 = (et_init_mz1_noisy - data_min_mz1[3])/(data_max_mz1[3]-data_min_mz1[3])
        rd_scaled_mzs = (rooting_depth - data_min_mz3[4])/(data_max_mz3[4]-data_min_mz3[4])
        lai_scaled_mzs = (lai_factors - data_min_mz3[5])/(data_max_mz3[5]-data_min_mz3[5])
        x0_init_mz2 = np.array(statesList_mz2[k:k+sequenceLength])
        u_init_mz2  = np.array(irrigationList_mz2[k:k+sequenceLength-1])
        x0_scaled_mz2 = (x0_init_mz2 - data_min_mz2[0])/(data_max_mz2[0]-data_min_mz2[0])
        x0_init_mz3 = np.array(statesList_mz3[k:k+sequenceLength])
        u_init_mz3  = np.array(irrigationList_mz3[k:k+sequenceLength-1])
        x0_scaled_mz3 = (x0_init_mz3 - data_min_mz3[0])/(data_max_mz3[0]-data_min_mz3[0])
    
        # #Without uncertainty
        # irrigationAmounts_mz1, irrigationAmounts_mz2, irrigationAmounts_mz3, irrigationDecisions, states_mz1, states_mz2, states_mz3 = SolveScheduler(allStates_mz1[k,:], 
        #     allStates_mz2[k,:],allStates_mz3[k,:], et_init_mz1[4:4+horizonLength_orig], kc_init_mz1[4:4+horizonLength_orig], 
        #     rooting_depth[4:4+horizonLength_orig], lai_factors[4:4+horizonLength_orig], rain[4:4+horizonLength_orig], x0_scaled_mz1, x0_scaled_mz2, 
        #     x0_scaled_mz3, u_init_mz1,  u_init_mz2, u_init_mz3, kc_scaled_mz1, et_scaled_mz1, rd_scaled_mzs, lai_scaled_mzs, rain)
    
        # ##With uncertainty
        # irrigationAmounts_mz1, irrigationAmounts_mz2, irrigationAmounts_mz3, irrigationDecisions, states_mz1, states_mz2, states_mz3 = SolveScheduler(allStates_mz1[k,:], 
        #     allStates_mz2[k,:],allStates_mz3[k,:], et_init_mz1_noisy[4:4+horizonLength_orig], kc_init_mz1[4:4+horizonLength_orig], 
        #     rooting_depth[4:4+horizonLength_orig], lai_factors[4:4+horizonLength_orig], rain[4:4+horizonLength_orig], x0_scaled_mz1, x0_scaled_mz2, 
        #     x0_scaled_mz3, u_init_mz1,  u_init_mz2, u_init_mz3, kc_scaled_mz1, et_scaled_mz1, rd_scaled_mzs, lai_scaled_mzs, rain_noisy)
    
        
        irrigationAmounts_mz1, irrigationAmounts_mz2, irrigationAmounts_mz3, irrigationDecisions, states_mz1, states_mz2, states_mz3 = SolveScheduler(allStates_mz1_est[k,:], 
            allStates_mz2_est[k,:],allStates_mz3_est[k,:], et_init_mz1_noisy[horizon_slice], kc_init_mz1[horizon_slice], 
            rooting_depth[horizon_slice], lai_factors[horizon_slice], rain[horizon_slice], x0_scaled_mz1, x0_scaled_mz2, 
            x0_scaled_mz3, u_init_mz1,  u_init_mz2, u_init_mz3, kc_scaled_mz1, et_scaled_mz1, rd_scaled_mzs, lai_scaled_mzs, rain_noisy)
        
        #For monitoring purposes
        print(f"The initial irrigation amount for the first management zone is {irrigationAmounts_mz1[0]}")
        print("\n\n")
        print(f"The initial irrigation amount for the second management zone is {irrigationAmounts_mz2[0]}")
        print("\n\n")
        print(f"The irrigation amount for the third management zone is {irrigationAmounts_mz3[0]}")
    
        states_mz1_plot = np.zeros(len(states_mz1)+1)
        states_mz1_plot[0] = x0_init_mz1[-1]
        states_mz1_plot[1:] = states_mz1
        states_mz2_plot = np.zeros(len(states_mz2)+1)
        states_mz2_plot[0] = x0_init_mz2[-1]
        states_mz2_plot[1:] = states_mz2
        states_mz3_plot = np.zeros(len(states_mz3)+1)
        states_mz3_plot[0] = x0_init_mz3[-1]
        states_mz3_plot[1:] = states_mz3
        
        fig, axs = plt.subplots(3, figsize=(15,15))
        axs[0].plot(states_mz1_plot)
        axs[0].plot(UpperZone_mz1*np.ones(len(states_mz1_plot)))
        axs[0].plot(LowerZone_mz1*np.ones(len(states_mz1_plot)))
        axs[1].plot(states_mz2_plot)
        axs[1].plot(UpperZone_mz2*np.ones(len(states_mz2_plot)))
        axs[1].plot(LowerZone_mz2*np.ones(len(states_mz2_plot)))
        axs[2].plot(states_mz3_plot)
        axs[2].plot(UpperZone_mz3*np.ones(len(states_mz3_plot)))
        axs[2].plot(LowerZone_mz3*np.ones(len(states_mz3_plot)))
        plt.close()
        
        irrigationList_mz1+=[irrigationAmounts_mz1[0]]
        prescribedIrrigation_closedLoop_mz1+=[irrigationAmounts_mz1[0]]
        stateTrajectory_openLoop_mz1+=[states_mz1[0]]
        irrigationList_mz2+=[irrigationAmounts_mz2[0]]
        prescribedIrrigation_closedLoop_mz2+=[irrigationAmounts_mz2[0]]
        stateTrajectory_openLoop_mz2+=[states_mz2[0]]
        irrigationList_mz3+=[irrigationAmounts_mz3[0]]
        prescribedIrrigation_closedLoop_mz3+=[irrigationAmounts_mz3[0]]
        stateTrajectory_openLoop_mz3+=[states_mz3[0]]
        irrigationDecisions_closedLoop+=[irrigationDecisions[0]]
        
        #Actual plant simulation (RE) - MZ 1
        x_currentSE_mz1  = allStates_mz1[k,:]
        x_currentSE_mz1_est = allStates_mz1_est[k,:]
        u_currentSE_mz1  = (irrigationAmounts_mz1[0] + rain_act[current_day_offset])/86400
        kc_currentSE_mz1 = kc_init_mz1[current_day_offset]
        et_currentSE_mz1 = et_init_mz1_act[current_day_offset]
        rd_currentSE_mz1 = rooting_depth[current_day_offset]
        lai_currentSE_mz1 = lai_factors[current_day_offset]
        currentStatesSE_mz1 =simulate_Richards_equation(x_currentSE_mz1, u_currentSE_mz1, et_currentSE_mz1, kc_currentSE_mz1, 4, rd_currentSE_mz1, lai_currentSE_mz1,soilPars_mz1)
        currentStatesSE_mz1_est =simulate_Richards_equation(x_currentSE_mz1_est, u_currentSE_mz1, et_currentSE_mz1, kc_currentSE_mz1, 4, rd_currentSE_mz1, lai_currentSE_mz1,soilPars_mz1)
        currentStatesSE_mz1_vol = obtain_soil_moisture(currentStatesSE_mz1, soilPars_mz1)
        rootzone_moist_mz1 = compute_root_zone_moisture(currentStatesSE_mz1_vol, rd_currentSE_mz1)
        # statesList_mz1+=[rootzone_moist_mz1]
        allStates_mz1[k+1,:] = currentStatesSE_mz1
    
        #Actual plant simulation (RE)- MZ 2 
        x_currentSE_mz2  = allStates_mz2[k,:]
        x_currentSE_mz2_est = allStates_mz2_est[k,:]
        u_currentSE_mz2  = (irrigationAmounts_mz2[0] + rain_act[current_day_offset])/86400
        kc_currentSE_mz2 = kc_init_mz1[current_day_offset]
        et_currentSE_mz2 = et_init_mz1_act[current_day_offset]
        currentStatesSE_mz2 =simulate_Richards_equation(x_currentSE_mz2, u_currentSE_mz2, et_currentSE_mz2, kc_currentSE_mz2, 4, rd_currentSE_mz1, lai_currentSE_mz1, soilPars_mz2)
        currentStatesSE_mz2_est =simulate_Richards_equation(x_currentSE_mz2_est, u_currentSE_mz2, et_currentSE_mz2, kc_currentSE_mz2, 4, rd_currentSE_mz1, lai_currentSE_mz1, soilPars_mz2)
        currentStatesSE_mz2_vol = obtain_soil_moisture(currentStatesSE_mz2, soilPars_mz2)
        rootzone_moist_mz2 = compute_root_zone_moisture(currentStatesSE_mz2_vol, rd_currentSE_mz1)
        # statesList_mz2+=[rootzone_moist_mz2]
        allStates_mz2[k+1,:] = currentStatesSE_mz2
        
        #Actual plant simulation (RE)- MZ 3
        x_currentSE_mz3 = allStates_mz3[k,:]
        x_currentSE_mz3_est = allStates_mz3_est[k,:]
        u_currentSE_mz3 = (irrigationAmounts_mz3[0] + rain_act[current_day_offset])/86400
        kc_currentSE_mz3 = kc_init_mz1[current_day_offset]
        et_currentSE_mz3 = et_init_mz1_act[current_day_offset]
        currentStatesSE_mz3 =simulate_Richards_equation(x_currentSE_mz3, u_currentSE_mz3, et_currentSE_mz3, kc_currentSE_mz3, 4, rd_currentSE_mz1, lai_currentSE_mz1, soilPars_mz3)
        currentStatesSE_mz3_est =simulate_Richards_equation(x_currentSE_mz3_est, u_currentSE_mz3, et_currentSE_mz3, kc_currentSE_mz3, 4, rd_currentSE_mz1, lai_currentSE_mz1, soilPars_mz3)
        currentStatesSE_mz3_vol = obtain_soil_moisture(currentStatesSE_mz3, soilPars_mz3)
        rootzone_moist_mz3 = compute_root_zone_moisture(currentStatesSE_mz3_vol, rd_currentSE_mz1)
        # statesList_mz3+=[rootzone_moist_mz3]
        allStates_mz3[k+1,:] = currentStatesSE_mz3
    
        #LSTM Model Simulation - MZ1    
        x_current_mz1 = np.array(statesList_mz1[k:k+sequenceLength])
        x_current_scaled_mz1 = (x_current_mz1 - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
        # u_current_mz1 = np.array(irrigationList_mz1[k:k+sequenceLength]) + rain_act[0:0+sequenceLength]
        u_current_mz1 = np.array(irrigationList_mz1[k:k+sequenceLength]) + rain_noisy[0:0+sequenceLength] #Rain with noisy should be used for the model simulation
        u_current_mz1 = u_current_mz1/86400
        u_current_scaled_mz1 = (u_current_mz1 - data_min_mz1[1])/(data_max_mz1[1] - data_min_mz1[1])
        #Ensure that the scaled irrigation rate ranges from 0 to 1
        u_current_scaled_mz1[u_current_scaled_mz1<0]=0
        u_current_scaled_mz1[u_current_scaled_mz1>1]=1
    
        kc_current_mz1 = kc_scaled_mz1[0:0+sequenceLength]
        et_current_mz1 = et_scaled_mz1[0:0+sequenceLength]
        rd_current_mz1 = rd_scaled_mzs[0:0+sequenceLength]
        lai_current_mz1 = lai_scaled_mzs[0:0+sequenceLength]
        currentState_mz1 = getRNNmodel_numpy_mz1(x_current_scaled_mz1, u_current_scaled_mz1, kc_current_mz1, et_current_mz1,rd_current_mz1, lai_current_mz1)
        currentState_unscaled_mz1 = (currentState_mz1 * (data_max_mz1[0] - data_min_mz1[0])) + data_min_mz1[0]
        # stateTrajectory_closedLoop_mz1+=[currentState_unscaled_mz1]
        
        #LSTM Model Simulation - MZ2  
        x_current_mz2 = np.array(statesList_mz2[k:k+sequenceLength])
        x_current_scaled_mz2 = (x_current_mz2 - data_min_mz2[0])/(data_max_mz2[0]-data_min_mz2[0])
        # u_current_mz2 = np.array(irrigationList_mz2[k:k+sequenceLength]) + rain_act[0:0+sequenceLength]
        u_current_mz2 = np.array(irrigationList_mz2[k:k+sequenceLength]) + rain_noisy[0:0+sequenceLength]
        u_current_mz2 = u_current_mz2/86400
        u_current_scaled_mz2 = (u_current_mz2 - data_min_mz2[1])/(data_max_mz2[1]-data_min_mz2[1])
        #Ensure that the scaled irrigation rate ranges from 0 to 1
        u_current_scaled_mz2[u_current_scaled_mz2<0]=0
        u_current_scaled_mz2[u_current_scaled_mz2>1]=1
        currentState_mz2 = getRNNmodel_numpy_mz2(x_current_scaled_mz2, u_current_scaled_mz2, kc_current_mz1, et_current_mz1,rd_current_mz1, lai_current_mz1)
        currentState_unscaled_mz2 = (currentState_mz2 * (data_max_mz2[0] - data_min_mz2[0])) + data_min_mz2[0]
        # stateTrajectory_closedLoop_mz2+=[currentState_unscaled_mz2]
    
        #LSTM Model Simulation - MZ3  
        x_current_mz3 = np.array(statesList_mz3[k:k+sequenceLength])
        x_current_scaled_mz3 = (x_current_mz3 - data_min_mz3[0])/(data_max_mz3[0]-data_min_mz3[0])
        # u_current_mz3 = np.array(irrigationList_mz3[k:k+sequenceLength]) + rain_act[0:0+sequenceLength]
        u_current_mz3 = np.array(irrigationList_mz3[k:k+sequenceLength]) + rain_noisy[0:0+sequenceLength]
        u_current_mz3 = u_current_mz3/86400
        u_current_scaled_mz3 = (u_current_mz3 - data_min_mz3[1])/(data_max_mz3[1]-data_min_mz3[1])
        #Ensure that the scaled irrigation rate ranges from 0 to 1
        u_current_scaled_mz3[u_current_scaled_mz3<0]=0
        u_current_scaled_mz3[u_current_scaled_mz3>1]=1
        currentState_mz3 = getRNNmodel_numpy_mz3(x_current_scaled_mz3, u_current_scaled_mz3, kc_current_mz1, et_current_mz1, rd_current_mz1, lai_current_mz1)
        currentState_unscaled_mz3 = (currentState_mz3 * (data_max_mz3[0] - data_min_mz3[0])) + data_min_mz3[0]
        
        # referenceEvap_all_mz1[4+k+1] = referenceEvap_all_act_mz1[4+k+1]
        # rain_all_predicted[4+k+1] = rain_all_act[4+k+1]
    
    
        if (k+1) % (k+1) == 0:
            # measurement_mz1=factor_mz1*rootzone_moist_mz1
            # measurement_mz2=factor_mz2*rootzone_moist_mz2
            # measurement_mz3=factor_mz3*rootzone_moist_mz3
    
            measurement_mz1=factor_mz1*surface_moisture_alg(currentStatesSE_mz1[10:], soilPars_mz1)
            measurement_mz2=factor_mz2*surface_moisture_alg(currentStatesSE_mz2[10:], soilPars_mz2)
            measurement_mz3=factor_mz3*surface_moisture_alg(currentStatesSE_mz3[10:], soilPars_mz3)
    
            x_currentSE_mz1_updated,covariance_mz1=discreteTimeEKF(x_currentSE_mz1_est[10:],currentStatesSE_mz1_est[10:],u_currentSE_mz1,kc_currentSE_mz1,
            et_currentSE_mz1,1.0,measurement_mz1,covariance_mz1,process_covariance,measurement_covariance,soilPars_mz1)
    
            x_currentSE_mz2_updated,covariance_mz2=discreteTimeEKF(x_currentSE_mz2_est[10:],currentStatesSE_mz2_est[10:],u_currentSE_mz2,kc_currentSE_mz1,
            et_currentSE_mz1,1.0,measurement_mz2, covariance_mz2,process_covariance,measurement_covariance,soilPars_mz2)
    
            x_currentSE_mz3_updated,covariance_mz3=discreteTimeEKF(x_currentSE_mz3_est[10:],currentStatesSE_mz3_est[10:],u_currentSE_mz3,kc_currentSE_mz1, 
            et_currentSE_mz1,1.0,measurement_mz3,covariance_mz3,process_covariance,measurement_covariance,soilPars_mz3)
    
            currentStatesSE_mz1_est=x_currentSE_mz1_updated
            currentStatesSE_mz2_est=x_currentSE_mz2_updated
            currentStatesSE_mz3_est=x_currentSE_mz3_updated
    
            currentStatesSE_mz1_est=np.concatenate((currentStatesSE_mz1[0:10],currentStatesSE_mz1_est))
            currentStatesSE_mz2_est=np.concatenate((currentStatesSE_mz2[0:10],currentStatesSE_mz2_est))
            currentStatesSE_mz3_est=np.concatenate((currentStatesSE_mz3[0:10],currentStatesSE_mz3_est))
    
            statesList_mz1+=[rz_moisture_alg(currentStatesSE_mz1_est[10:],soilPars_mz1)]
            statesList_mz2+=[rz_moisture_alg(currentStatesSE_mz2_est[10:],soilPars_mz2)]
            statesList_mz3+=[rz_moisture_alg(currentStatesSE_mz3_est[10:],soilPars_mz3)]
    
        else:
            statesList_mz1+=[currentState_unscaled_mz1]
            statesList_mz2+=[currentState_unscaled_mz2]
            statesList_mz3+=[currentState_unscaled_mz3]
            pass
        
        allStates_mz1_est[k+1,:] = currentStatesSE_mz1_est
        allStates_mz2_est[k+1,:] = currentStatesSE_mz2_est
        allStates_mz3_est[k+1,:] = currentStatesSE_mz3_est
    
        stateTrajectory_closedLoop_mz1+=[rootzone_moist_mz1]
        stateTrajectory_closedLoop_mz2+=[rootzone_moist_mz2]
        stateTrajectory_closedLoop_mz3+=[rootzone_moist_mz3]
    
    
        pass
    
    #Post-processing and visualization
    u_prescribed_mz1 = np.array(prescribedIrrigation_closedLoop_mz1)*1000
    u_prescribed_mz2 = np.array(prescribedIrrigation_closedLoop_mz2)*1000
    u_prescribed_mz3 = np.array(prescribedIrrigation_closedLoop_mz3)*1000  
        
    def ObtainPressureHead(volMoisture, soilPars):
        pressure_head = []
        for i in range(len(volMoisture)):
            pressure_head.append(pressureHead(volMoisture[i], soilPars))
            pass
        return np.array(pressure_head)
    
    #Obtain the arrays for the time
    timeStamps = np.arange(1, simulationPeriod+2)
    timeStamps_input = np.arange(1, simulationPeriod+1)
    
    fig, axs = plt.subplots(3,2, figsize=(15,15))
    axs[0,0].bar(timeStamps_input, -u_prescribed_mz1, width = 1.2, facecolor=None)
    rain_plot_start = sequenceLength - 1
    axs[0,0].bar(timeStamps_input, _rain_act[rain_plot_start:rain_plot_start+len(u_prescribed_mz1)], color='red', width = 1.0)
    axs[0,0].legend(['Proposed', 'Rain'])
    axs[0,0].set_title('Irrigation -  MZ 1 ')
    axs[0,0].set_ylabel('Irrigation (mm/day)')
    axs[0,0].set_xlabel('Time (days)')
    axs[0,0].set_xlim(xmin=1)
    
    axs[0,1].plot(timeStamps, stateTrajectory_closedLoop_mz1, color='blue', linewidth=1.8)
    axs[0,1].plot(timeStamps, UpperZone_mz1*np.ones(len(stateTrajectory_openLoop_mz1)), color='darkcyan', linestyle = 'dashdot', linewidth=2.5)
    axs[0,1].plot(timeStamps, LowerZone_mz1*np.ones(len(stateTrajectory_openLoop_mz1)), color='darkmagenta', linestyle = 'dashdot', linewidth=2.5)
    axs[0,1].plot(timeStamps,0.12*np.ones(len(stateTrajectory_openLoop_mz1)), color='red', linestyle='dashdot', linewidth=2.5)
    axs[0,1].legend(['state Trajectory', 'FC', 'Threshold', 'PWP'])
    axs[0,1].set_title('Root Zone Soil Moisture - MZ 1')
    axs[0,1].set_ylabel('Volumetric moisture')
    axs[0,1].set_xlabel('Time (days)')
    axs[0,1].set_xlim(xmin=1)
    
    axs[1,0].bar(timeStamps_input, -u_prescribed_mz2, width = 1.2)
    axs[1,0].bar(timeStamps_input, _rain_act[rain_plot_start:rain_plot_start+len(u_prescribed_mz1)], color='red', width = 1.0)
    axs[1,0].legend(['Proposed', 'Rain'])
    axs[1,0].set_title('Irrigation -  MZ 2')
    axs[1,0].set_ylabel('Irrigation (mm/day)')
    axs[1,0].set_xlabel('Time (days)')
    axs[1,0].set_xlim(xmin=1)
    
    axs[1,1].plot(timeStamps, stateTrajectory_closedLoop_mz2, color='blue', linewidth=1.8)
    axs[1,1].plot(timeStamps, UpperZone_mz2*np.ones(len(stateTrajectory_openLoop_mz2)), color='darkcyan', linestyle = 'dashdot', linewidth=2.5)
    axs[1,1].plot(timeStamps, LowerZone_mz2*np.ones(len(stateTrajectory_openLoop_mz2)), color='darkmagenta', linestyle = 'dashdot', linewidth=2.5)
    axs[1,1].plot(timeStamps, 0.12*np.ones(len(stateTrajectory_openLoop_mz1)), color='red', linestyle='dashdot', linewidth=2.5)
    axs[1,1].legend(['state Trajectory', 'FC', 'Threshold', 'PWP'])
    axs[1,1].set_title('Root Zone Soil Moisture -  MZ 2')
    axs[1,1].set_ylabel('Volumetric moisture')
    axs[1,1].set_xlabel('Time (days)')
    axs[1,1].set_xlim(xmin=1)
    
    axs[2,0].bar(timeStamps_input, -u_prescribed_mz3, width = 1.2)
    axs[2,0].bar(timeStamps_input, _rain_act[rain_plot_start:rain_plot_start+len(u_prescribed_mz1)], color='red', width = 1.0)
    axs[2,0].legend(['Proposed', 'Rain'])
    axs[2,0].set_title('Irrigation - MZ 3')
    axs[2,0].set_ylabel('Irrigation (mm/day)')
    axs[2,0].set_xlabel('Time (days)')
    axs[2,0].set_xlim(xmin=1)
    
    axs[2,1].plot(timeStamps, stateTrajectory_closedLoop_mz3, color='blue', linewidth=1.8)
    axs[2,1].plot(timeStamps, UpperZone_mz3*np.ones(len(stateTrajectory_openLoop_mz3)), color='darkcyan', linestyle = 'dashdot', linewidth=2.5)
    axs[2,1].plot(timeStamps, LowerZone_mz3*np.ones(len(stateTrajectory_openLoop_mz3)), color='darkmagenta', linestyle = 'dashdot', linewidth=2.5)
    axs[2,1].plot(timeStamps, 0.16*np.ones(len(stateTrajectory_openLoop_mz1)), color='red',linestyle='dashdot', linewidth=2.5)
    axs[2,1].legend(['state Trajectory', 'FC', 'Threshold', 'PWP'])
    axs[2,1].set_title('Root Zone Soil Moisture -  MZ 3')
    axs[2,1].set_ylabel('Volumetric moisture')
    axs[2,1].set_xlabel('Time (days)')
    axs[2,1].set_xlim(xmin=1)
    fig.tight_layout()
    fig.savefig("./results_cl-scmarl/Closed_loop.pdf")
    np.savetxt('./results_cl-scmarl/StateCL_mz1.txt', stateTrajectory_closedLoop_mz1)
    np.savetxt('./results_cl-scmarl/InputCL_mz1.txt', u_prescribed_mz1)
    np.savetxt('./results_cl-scmarl/StateCL_mz2.txt', stateTrajectory_closedLoop_mz2)
    np.savetxt('./results_cl-scmarl/InputCL_mz2.txt', u_prescribed_mz2)
    np.savetxt('./results_cl-scmarl/StateCL_mz3.txt', stateTrajectory_closedLoop_mz3)
    np.savetxt('./results_cl-scmarl/InputCL_mz3.txt', u_prescribed_mz3)
    
    #calculate the yield
    yield_ = evaluate_yield()
    total_irrigation_rate = -1*(np.sum(u_prescribed_mz1) + np.sum(u_prescribed_mz2) + np.sum(u_prescribed_mz3))/1000
    IWUE = yield_/total_irrigation_rate 

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()

