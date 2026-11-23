import sys
sys.path.append("../common/actual_field")
import numpy as np
from crop_coefficient_calculation import generate_Kc
from van_Genuchten_relations import volumetricMoisture

from parameters_mpc import *
# Load the  soil parameters for the MZs
soilPars_mz1 = ManagementZone_1()
soilPars_mz2 = ManagementZone_2()
soilPars_mz3 = ManagementZone_3()


# Lower and upper bounds of the target zone for each MZ
LowerZone_mz1=0.200
UpperZone_mz1=0.281
LowerZone_mz2=0.200
UpperZone_mz2=0.281  
LowerZone_mz3=0.230
UpperZone_mz3=0.301 
    
# pwp_mz1 = 0.12
# pwp_mz2 = 0.12
# pwp_mz3 = 0.16
    
pwp_mz1 = volumetricMoisture(-152.85, soilPars_mz1)
pwp_mz2 = volumetricMoisture(-152.85, soilPars_mz2)
pwp_mz3 = volumetricMoisture(-152.85, soilPars_mz3)
        
anaerobic_point_mz1 = volumetricMoisture(-0.1, soilPars_mz1)
anaerobic_point_mz2 = volumetricMoisture(-0.1, soilPars_mz2)
anaerobic_point_mz3 = volumetricMoisture(-0.1, soilPars_mz3)

maximal_potential_yield = 8.8
yield_response_factor = 1.15

def evaluate_water_stress_factor_mz1(current_vol_moisture):
    if current_vol_moisture <= pwp_mz1:
        return 0
    elif pwp_mz1 < current_vol_moisture <= LowerZone_mz1:
        return (current_vol_moisture - pwp_mz1)/(LowerZone_mz1 - pwp_mz1)

    elif LowerZone_mz1 < current_vol_moisture < UpperZone_mz1:
        return 1

    elif UpperZone_mz1 <= current_vol_moisture <= anaerobic_point_mz1:
        return (current_vol_moisture - UpperZone_mz1)/(anaerobic_point_mz1 - UpperZone_mz1)
    else:
        return 0

def evaluate_water_stress_factor_mz2(current_vol_moisture):
    if current_vol_moisture <= pwp_mz2:
        return 0
    elif pwp_mz2 < current_vol_moisture <= LowerZone_mz2:
        return (current_vol_moisture - pwp_mz2)/(LowerZone_mz2 - pwp_mz2)

    elif LowerZone_mz2 < current_vol_moisture < UpperZone_mz2:
        return 1
    elif UpperZone_mz2 <= current_vol_moisture <= anaerobic_point_mz2:
        return (current_vol_moisture - UpperZone_mz2)/(anaerobic_point_mz2 - UpperZone_mz2)
    else:
        return 0

def evaluate_water_stress_factor_mz3(current_vol_moisture):
    if current_vol_moisture <= pwp_mz3:
        return 0
    elif pwp_mz3 < current_vol_moisture <= LowerZone_mz3:
        return (current_vol_moisture - pwp_mz3)/(LowerZone_mz3 - pwp_mz3)

    elif LowerZone_mz3 < current_vol_moisture < UpperZone_mz3:
        return 1
    elif UpperZone_mz3 <= current_vol_moisture <= anaerobic_point_mz3:
        return (current_vol_moisture - UpperZone_mz3)/(anaerobic_point_mz3 - UpperZone_mz3)
    else:
        return 0


def evaluate_yield():

    # print(pwp_mz1)
    # print(pwp_mz2)
    # print(pwp_mz3)
    
    # print(anaerobic_point_mz1)
    # print(anaerobic_point_mz2)
    # print(anaerobic_point_mz3)
    
    # Load the weather conditions
    _average_temps =  np.loadtxt("../common/weather_data/dAVG_TEMP.txt")
    _crop_coeff = generate_Kc(_average_temps)
    _rain = np.loadtxt("../common/weather_data/dRAIN.txt")
    _ref_Evap = np.loadtxt("../common/weather_data/dPET.txt")
    
    for i in range(len(_ref_Evap)):
        if _ref_Evap[i] < 1.04:
            _ref_Evap[i] = 1.04
        if _ref_Evap[i] > 9.0:
            _ref_Evap[i] = 9.0
        pass
    # Load the state trajectory over the simukation period
    state_traj_mz1 = np.loadtxt('./results_cl-scmarl/StateCL_mz1.txt')
    state_traj_mz2 = np.loadtxt('./results_cl-scmarl/StateCL_mz2.txt')
    state_traj_mz3 = np.loadtxt('./results_cl-scmarl/StateCL_mz3.txt')
    
    water_stress_factor_mz1 = []
    water_stress_factor_mz2 = []
    water_stress_factor_mz3 = []
    
    maximum_ET_mz1 = []
    maximum_ET_mz2 = []
    maximum_ET_mz3 = []
    
    seasonal_ET_mz1 = []
    seasonal_ET_mz2 = []
    seasonal_ET_mz3 = []
    
    
    for i in range(len(state_traj_mz1)):
        water_stress_factor_mz1.append(
            evaluate_water_stress_factor_mz1(state_traj_mz1[i]))
        water_stress_factor_mz2.append(
            evaluate_water_stress_factor_mz2(state_traj_mz2[i]))
        water_stress_factor_mz3.append(
            evaluate_water_stress_factor_mz3(state_traj_mz3[i]))
    
        maximum_ET_mz1.append(_crop_coeff[i]*_ref_Evap[i])
        maximum_ET_mz2.append(_crop_coeff[i]*_ref_Evap[i])
        maximum_ET_mz3.append(_crop_coeff[i]*_ref_Evap[i])
    
        seasonal_ET_mz1.append(evaluate_water_stress_factor_mz1(
            state_traj_mz1[i])*_crop_coeff[i]*_ref_Evap[i])
        seasonal_ET_mz2.append(evaluate_water_stress_factor_mz2(
            state_traj_mz2[i])*_crop_coeff[i]*_ref_Evap[i])
        seasonal_ET_mz3.append(evaluate_water_stress_factor_mz3(
            state_traj_mz3[i])*_crop_coeff[i]*_ref_Evap[i])
        pass
    
    water_stress_factor_mz1 = np.array(water_stress_factor_mz1)
    water_stress_factor_mz2 = np.array(water_stress_factor_mz2)
    water_stress_factor_mz3 = np.array(water_stress_factor_mz3)
    
    pred_yield_mz1 = (1 - yield_response_factor + yield_response_factor *
                      (sum(np.array(seasonal_ET_mz1))/sum(np.array(maximum_ET_mz1))))*maximal_potential_yield
    pred_yield_mz2 = (1 - yield_response_factor + yield_response_factor *
                      (sum(np.array(seasonal_ET_mz2))/sum(np.array(maximum_ET_mz2))))*maximal_potential_yield
    pred_yield_mz3 = (1 - yield_response_factor + yield_response_factor *
                      (sum(np.array(seasonal_ET_mz3))/sum(np.array(maximum_ET_mz3))))*maximal_potential_yield
    
    # print(pred_yield_mz1)
    # print(pred_yield_mz2)
    # print(pred_yield_mz3)

    average_yield = (pred_yield_mz1 + pred_yield_mz2 + pred_yield_mz3)/3
    average_yield = pred_yield_mz1
    return average_yield