import sys
sys.path.append("../common/actual_field")
from model_simulation import * 
import numpy as np 
from tensorforce.agents import Agent
from utils_agent import *
from temporal_spatial_parameters_field import *
totalDepth, axialNodes, totalNodes= spatialVariables_1D_act()
lower_zone_mz1 = 0.176
upper_zone_mz1 = 0.280
lower_zone_mz2 = 0.176
upper_zone_mz2 = 0.280
lower_zone_mz3 = 0.209
upper_zone_mz3 = 0.300


def compute_rootZone_moisture(current_state, rooting_depth):
        if rooting_depth == 0.50:
            rootZone_vol_moist = 0.10*((1/6)*(current_state[10] + current_state[11] + current_state[12] + current_state[13] + current_state[14] + current_state[15])) +\
                0.20*((1/6)*(current_state[15] + current_state[16] + current_state[17] + current_state[18] + current_state[19] + current_state[20])) +\
                0.30*((1/6)*(current_state[20] + current_state[21] + current_state[22] + current_state[23] + current_state[24] + current_state[25])) + \
                0.40*((1/6)*(current_state[25] + current_state[26] + current_state[27] + current_state[28] + current_state[29] + current_state[30]))
        else:
            rootZone_vol_moist = 0.10*((1/6)*(current_state[0] + current_state[1] + current_state[2] + current_state[3] + current_state[4] + current_state[5]))  + \
                0.20*((1/6)*(current_state[5] + current_state[6] + current_state[7] + current_state[8] + current_state[9] + current_state[10])) + \
                0.30*((1/11)*(current_state[10] + current_state[11] + current_state[12] + current_state[13] + current_state[14] + current_state[15] + current_state[16] + current_state[17] + current_state[18] + current_state[19] + current_state[20])) + \
                0.40*((1/11)*(current_state[20] + current_state[21] + current_state[22] + current_state[23] + current_state[24] + current_state[25] + current_state[26] + current_state[27] + current_state[28] + current_state[29] + current_state[30]))
        return rootZone_vol_moist

_agent_name = 'Scheduler_agent -5-5'
_agent_mz1 = Agent.load("./trained_rl_agents_b/hybrid_ppo_agents/agent_mz1/", filename=_agent_name)
_agent_mz2 = Agent.load("./trained_rl_agents_b/hybrid_ppo_agents/agent_mz2/", filename=_agent_name)
_agent_mz3 = Agent.load("./trained_rl_agents_b/hybrid_ppo_agents/agent_mz3/", filename=_agent_name)


def obtain_discrete_actions_mz1(current_state, refEvap_sequence, cropCoeff_sequence, rootingDepth_sequence, lai_sequence, rain_sequence, prediction_horizon, soil_pars):
    agent_loaded = _agent_mz1
    internals = agent_loaded.initial_internals()
    #Convert the current state in pressure head to volumetric moisture
    initial_state_vol = volMoistureAllNodes_1D(current_state, soil_pars)
    state_init = initial_state_vol
    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:,0] = refEvap_sequence*(86400*1000) #Convert to mm/day
    uncontrolled_inputs[:,1] = cropCoeff_sequence
    uncontrolled_inputs[:,2] = rootingDepth_sequence
    uncontrolled_inputs[:,3] = lai_sequence
    stateArray = np.zeros((prediction_horizon+1, totalNodes))
    stateArray_rz = np.zeros(prediction_horizon+1)    
    actionsArray_discrete = np.zeros(prediction_horizon)
    actionsArray_continuous = np.zeros(prediction_horizon)
    stateArray[0,:] = state_init
    stateArray_rz[0] = compute_rootZone_moisture(state_init, rootingDepth_sequence[0]) # The initial rooting depth is 25 cm
    for i in range(prediction_horizon):
        states = np.concatenate((stateArray[i,:], uncontrolled_inputs[i,:])) 
        if abs(rain_sequence[i]) > 0.01: #0.01 : # Will have to play with this value going forward 
            actionsArray_continuous[i] = 0 
            actionsArray_discrete[i] = 0 
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)
        else:
            actions, internals = agent_loaded.act(states=states, internals=internals, independent=True)
            actionsArray_continuous[i] = actions["irrigationRate"]*actions["irrigationDecision"]
            actionsArray_discrete[i] = actions["irrigationDecision"]
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (actions["irrigationRate"]*actions["irrigationDecision"] + rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)
        
        stateArray_rz[i+1] = compute_rootZone_moisture(next_state, rootingDepth_sequence[i])
        stateArray[i+1,:] = next_state
    pass 
    GenerateTrajectories_OL(prediction_horizon, -actionsArray_continuous,stateArray_rz,lower_zone_mz1, upper_zone_mz1)
    return actionsArray_discrete, actionsArray_continuous, stateArray_rz 

def obtain_discrete_actions_mz2(current_state, refEvap_sequence, cropCoeff_sequence, rootingDepth_sequence, lai_sequence, rain_sequence, prediction_horizon, soil_pars):
    agent_loaded = _agent_mz2
    internals = agent_loaded.initial_internals()
    #Convert the current state in pressure head to volumetric moisture
    initial_state_vol = volMoistureAllNodes_1D(current_state, soil_pars)
    state_init = initial_state_vol
    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:,0] = refEvap_sequence*(86400*1000)
    uncontrolled_inputs[:,1] = cropCoeff_sequence
    uncontrolled_inputs[:,2] = rootingDepth_sequence
    uncontrolled_inputs[:,3] = lai_sequence
    stateArray = np.zeros((prediction_horizon+1, totalNodes))
    stateArray_rz = np.zeros(prediction_horizon+1)    
    actionsArray_discrete = np.zeros(prediction_horizon)
    actionsArray_continuous = np.zeros(prediction_horizon)
    stateArray[0,:] = state_init
    stateArray_rz[0] = compute_rootZone_moisture(state_init, rootingDepth_sequence[0])
    for i in range(prediction_horizon):
        states = np.concatenate((stateArray[i,:], uncontrolled_inputs[i,:])) 
        if abs(rain_sequence[i]) > 0.01: #0.01: # Will have to play with this value going forward
            actionsArray_continuous[i] = 0 
            actionsArray_discrete[i] = 0 
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (0 + rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)
        else:
            actions, internals = agent_loaded.act(states=states, internals=internals, independent=True)
            actionsArray_continuous[i] = actions["irrigationRate"]*actions["irrigationDecision"]
            actionsArray_discrete[i] = actions["irrigationDecision"]
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (actions["irrigationRate"]*actions["irrigationDecision"] + rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)

        stateArray_rz[i+1] = compute_rootZone_moisture(next_state, rootingDepth_sequence[i])
        stateArray[i+1,:] = next_state
    pass 
    GenerateTrajectories_OL(prediction_horizon, -actionsArray_continuous,stateArray_rz,lower_zone_mz2, upper_zone_mz3)
    return actionsArray_discrete, actionsArray_continuous, stateArray_rz 

def obtain_discrete_actions_mz3(current_state, refEvap_sequence, cropCoeff_sequence, rootingDepth_sequence, lai_sequence, rain_sequence, prediction_horizon, soil_pars):
    agent_loaded = _agent_mz3
    internals = agent_loaded.initial_internals()
    #Convert the current state in pressure head to volumetric moisture
    initial_state_vol = volMoistureAllNodes_1D(current_state, soil_pars)
    state_init = initial_state_vol
    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:,0] = refEvap_sequence*(86400*1000)
    uncontrolled_inputs[:,1] = cropCoeff_sequence
    uncontrolled_inputs[:,2] = rootingDepth_sequence
    uncontrolled_inputs[:,3] = lai_sequence
    stateArray = np.zeros((prediction_horizon+1, totalNodes))
    stateArray_rz = np.zeros(prediction_horizon+1)    
    actionsArray_discrete = np.zeros(prediction_horizon)
    actionsArray_continuous = np.zeros(prediction_horizon)
    stateArray[0,:] = state_init
    stateArray_rz[0] = compute_rootZone_moisture(state_init, rootingDepth_sequence[0])
    for i in range(prediction_horizon):
        states = np.concatenate((stateArray[i,:], uncontrolled_inputs[i,:])) 
        if abs(rain_sequence[i]) > 0.01: #0.01:  #Will have to play with the value going forward
            actionsArray_continuous[i] = 0 
            actionsArray_discrete[i] = 0 
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (0 + rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)
        else:
            actions, internals = agent_loaded.act(states=states, internals=internals, independent=True)
            actionsArray_continuous[i] = actions["irrigationRate"]*actions["irrigationDecision"]
            actionsArray_discrete[i] = actions["irrigationDecision"]
            current_state=stateArray[i,:]
            next_state = SimulateModel(current_state, (actions["irrigationRate"]*actions["irrigationDecision"] + rain_sequence[i]),
                               uncontrolled_inputs[i,0], uncontrolled_inputs[i,1], uncontrolled_inputs[i,2], uncontrolled_inputs[i,3], soil_pars)

        stateArray_rz[i+1] = compute_rootZone_moisture(next_state, rootingDepth_sequence[i])
        stateArray[i+1,:] = next_state
    pass 
    GenerateTrajectories_OL(prediction_horizon, -actionsArray_continuous,stateArray_rz,lower_zone_mz3, upper_zone_mz3)
    return actionsArray_discrete, actionsArray_continuous, stateArray_rz 
