import sys 
sys.path.append('../common/environments/')
from environments import SimulateModel_UA, SimulateModel_LA_a, SimulateModel_LA_b, SimulateModel_LA_c
import numpy as np
from tensorforce.agents import Agent
from utils_scmarl import GenerateTrajectories
from temporal_spatial_scheduler import spatialVariables_1D, pars_mz1,pars_mz2, pars_mz3
from van_Genuchten_scheduler import volumetricMoisture
totalDepth, axialNodes, totalNodes = spatialVariables_1D() 

no_of_mzs = 3
# Load the parameters of the various management zones
soilPars_mz1 = pars_mz1()
soilPars_mz2 = pars_mz2()
soilPars_mz3 = pars_mz3()

#User defined function that computes the root zone soil moisture based on the spatial distribution of soil moisture in the column
def compute_root_zone_moisture(axial_state):
    rootZone_vol_moist = \
        0.10*((1/6)*(axial_state[0] + axial_state[1] + axial_state[2] + axial_state[3] + axial_state[4] + axial_state[5])) + \
        0.20*((1/6)*(axial_state[5] + axial_state[6] + axial_state[7] + axial_state[8] + axial_state[9] + axial_state[10])) + \
        0.30*((1/6)*(axial_state[10] + axial_state[11] + axial_state[12] + axial_state[13] + axial_state[14] + axial_state[15])) + \
        0.40*((1/6)*(axial_state[15] + axial_state[16] + axial_state[17] + axial_state[18] + axial_state[19] + axial_state[20]))
    return rootZone_vol_moist

def evaluate_scheduler_head(states_mz1,states_mz2,states_mz3,refEvap_sequence,cropCoeff_sequence,lai_sequence,rain_sequence,prediction_horizon):
    
    #convert the pressure head to volumetric soil moisture
    states_init_volMoist_mz1 = volumetricMoisture(states_mz1, soilPars_mz1)
    states_init_volMoist_mz2 = volumetricMoisture(states_mz2, soilPars_mz2)
    states_init_volMoist_mz3 = volumetricMoisture(states_mz3, soilPars_mz3)

    # Consider only the top 50cm of the soil column
    states_init_volMoist_mz1 =  states_init_volMoist_mz1[10:]
    states_init_volMoist_mz2 =  states_init_volMoist_mz2[10:]
    states_init_volMoist_mz3 =  states_init_volMoist_mz3[10:]


    uncontrolled_inputs = np.zeros((prediction_horizon, 4))
    uncontrolled_inputs[:,0] = refEvap_sequence*(86400*1000) #Convert to mm/day
    uncontrolled_inputs[:,1] = cropCoeff_sequence
    uncontrolled_inputs[:,2] = lai_sequence
    uncontrolled_inputs[:,3] = np.abs(rain_sequence)

    #Load the agents
    agent_name_a = 'Scheduler_agent_a - 12-2'
    agent_loaded_a = Agent.load("./trained_rl_agents_b/scmarl_agents/agents/",filename=agent_name_a)
    internals_ua = agent_loaded_a.initial_internals()

    agent_name_b = 'Scheduler_agent_b - 12-2'
    agent_loaded_b = Agent.load("./trained_rl_agents_b/scmarl_agents/agents/",filename=agent_name_b)
    internals_b = agent_loaded_b.initial_internals()

    agent_name_c = 'Scheduler_agent_c - 12-2'
    agent_loaded_c = Agent.load("./trained_rl_agents_b/scmarl_agents/agents/",filename=agent_name_c)
    internals_c = agent_loaded_c.initial_internals()

    agent_name_d = 'Scheduler_agent_d - 12-2'
    agent_loaded_d = Agent.load("./trained_rl_agents_b/scmarl_agents/agents/",filename=agent_name_d)
    internals_d = agent_loaded_d.initial_internals()

    
    #Arrays to store the results
    stateArray_ua=np.zeros((prediction_horizon+1, int(3*axialNodes)))
    actionsArray_ua=np.zeros(prediction_horizon) 

    stateArray_b=np.zeros((prediction_horizon+1, axialNodes))
    stateArray_rz_b=np.zeros(prediction_horizon+1)
    actionsArray_b=np.zeros(prediction_horizon)

    stateArray_c=np.zeros((prediction_horizon+1, axialNodes))
    stateArray_rz_c=np.zeros(prediction_horizon+1)
    actionsArray_c=np.zeros(prediction_horizon) 

    stateArray_d=np.zeros((prediction_horizon+1, axialNodes))
    stateArray_rz_d=np.zeros(prediction_horizon+1)
    actionsArray_d=np.zeros(prediction_horizon)

    #Define the bounds of the zone, MAD of 50% is used in this simulation
    lower_zone_mz1=0.200#0.200
    upper_zone_mz1=0.280
    lower_zone_mz3=0.230#0.230
    upper_zone_mz3=0.300

    vol_init_a = np.zeros(3*axialNodes)
    vol_init_a[0*axialNodes:1*axialNodes]=states_init_volMoist_mz1
    vol_init_a[1*axialNodes:2*axialNodes]=states_init_volMoist_mz2
    vol_init_a[2*axialNodes:3*axialNodes]=states_init_volMoist_mz3

    stateArray_ua[0,:]=vol_init_a 
    stateArray_b[0,:]=states_init_volMoist_mz1 
    stateArray_c[0,:]=states_init_volMoist_mz2
    stateArray_d[0,:]=states_init_volMoist_mz3

    stateArray_rz_b[0]=compute_root_zone_moisture(states_init_volMoist_mz1)
    stateArray_rz_c[0]=compute_root_zone_moisture(states_init_volMoist_mz2)
    stateArray_rz_d[0]=compute_root_zone_moisture(states_init_volMoist_mz3)

    for i in range(prediction_horizon):
        states_ua=np.concatenate((stateArray_ua[i,:],uncontrolled_inputs[i,:]))
        states_b=np.concatenate((stateArray_b[i,:],uncontrolled_inputs[i,:]))
        states_c=np.concatenate((stateArray_c[i,:],uncontrolled_inputs[i,:]))
        states_d=np.concatenate((stateArray_d[i,:],uncontrolled_inputs[i,:]))

        if (abs(rain_sequence[i])*1000) > 0.00: #10.00: #0.000: # use 0 for now and then use 10 mm/day later 
            # Do not operate the scheduler for a rain depth above 5 mm/day
            actionsArray_ua[i]=0.0
            actionsArray_b[i]=0.0
            actionsArray_c[i]=0.0
            actionsArray_d[i]=0.0

            current_state_ua=states_ua
            current_state_b=states_b
            current_state_c=states_c
            current_state_d=states_d

            action_all=np.zeros(no_of_mzs)
            action_all[0]= -uncontrolled_inputs[i,3]
            action_all[1]= -uncontrolled_inputs[i,3]
            action_all[2]= -uncontrolled_inputs[i,3]


            next_state_ua=SimulateModel_UA(current_state_ua[0:3*axialNodes],action_all,uncontrolled_inputs[i,0],
                                        uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            next_state_b=SimulateModel_LA_a(current_state_b[0:axialNodes],-uncontrolled_inputs[i,3],uncontrolled_inputs[i,0],
                                        uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)    
            next_state_c=SimulateModel_LA_b(current_state_c[0:axialNodes],-uncontrolled_inputs[i,3],uncontrolled_inputs[i,0],
                                        uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            next_state_d=SimulateModel_LA_c(current_state_d[0:axialNodes],-uncontrolled_inputs[i,3],uncontrolled_inputs[i,0],
                                        uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            
        else:
            action_ua,internals_ua=agent_loaded_a.act(states=states_ua,internals=internals_ua,independent=True)
            current_state_ua=states_ua
            current_state_b=states_b
            current_state_c=states_c
            current_state_d=states_d

            #The discrete action is an observation to the lower agents
            states_b = np.concatenate((states_b, np.array([action_ua])))
            states_c = np.concatenate((states_c, np.array([action_ua])))
            states_d = np.concatenate((states_d, np.array([action_ua])))
            
            action_b,internals_b=agent_loaded_b.act(states=states_b,internals=internals_b,independent=True)
            action_c,internals_c=agent_loaded_c.act(states=states_c,internals=internals_c,independent=True)
            action_d,internals_d=agent_loaded_d.act(states=states_d,internals=internals_d,independent=True)


            if abs(action_b) > 0.040:
                action_b = -0.040
                pass

            if abs(action_c) > 0.040:
                action_c = -0.040
                pass

            if abs(action_d) > 0.040:
                action_d = -0.040
                pass


            action_b=action_ua*action_b
            action_c=action_ua*action_c
            action_d=action_ua*action_d

            action_all=np.zeros(no_of_mzs)
            action_all[0]=action_b - uncontrolled_inputs[i,3] 
            action_all[1]=action_c - uncontrolled_inputs[i,3]
            action_all[2]=action_d - uncontrolled_inputs[i,3]


            action_b_mod = action_b - uncontrolled_inputs[i,3]
            action_c_mod = action_c - uncontrolled_inputs[i,3]
            action_d_mod = action_d - uncontrolled_inputs[i,3]

            next_state_ua=SimulateModel_UA(current_state_ua[0:3*axialNodes],action_all,uncontrolled_inputs[i,0],uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            next_state_b=SimulateModel_LA_a(current_state_b[0:axialNodes],action_b_mod,uncontrolled_inputs[i,0],uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            next_state_c=SimulateModel_LA_b(current_state_c[0:axialNodes],action_c_mod,uncontrolled_inputs[i,0],uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            next_state_d=SimulateModel_LA_c(current_state_d[0:axialNodes],action_d_mod,uncontrolled_inputs[i,0],uncontrolled_inputs[i,1],uncontrolled_inputs[i,2],0)
            
            actionsArray_ua[i]=action_ua
            actionsArray_b[i]=action_b
            actionsArray_c[i]=action_c
            actionsArray_d[i]=action_d

        stateArray_ua[i+1,:]=next_state_ua
        stateArray_b[i+1,:]=next_state_b
        stateArray_c[i+1,:]=next_state_c
        stateArray_d[i+1,:]=next_state_d

        stateArray_rz_b[i+1]=compute_root_zone_moisture(next_state_b)
        stateArray_rz_c[i+1]=compute_root_zone_moisture(next_state_c)
        stateArray_rz_d[i+1]=compute_root_zone_moisture(next_state_d)

    return stateArray_rz_b, stateArray_rz_c, stateArray_rz_d, actionsArray_ua, actionsArray_b, actionsArray_c, actionsArray_d


