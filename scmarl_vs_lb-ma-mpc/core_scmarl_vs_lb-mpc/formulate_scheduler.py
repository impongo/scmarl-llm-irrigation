#an mad of 50% is used in this simulation
import sys
sys.path.append("trained_lstm_models")
import casadi as cs
from lstm_mz1 import optimizeInputModel_automated_mz1
from lstm_mz2 import optimizeInputModel_automated_mz2
from lstm_mz3 import optimizeInputModel_automated_mz3
import numpy as np
interPoints = 14 #20 #Prediction horizon
sequenceLength = 5
data_min_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
data_min_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz2_vrd_lai.txt')
data_max_mz2 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz2_vrd_lai.txt')
data_min_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz3_vrd_lai.txt')
data_max_mz3 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz3_vrd_lai.txt')

#Lower and upper bounds of the target zone for each MZ. Upper bound of the target zones have been modified. A systematic approach will be required to determine the new upper bound.
LowerZone_mz1 = 0.200 
UpperZone_mz1 = 0.280 #0.280 
LowerZone_mz2 = 0.200 
UpperZone_mz2 = 0.280 #0.280 
LowerZone_mz3 = 0.230 
UpperZone_mz3 = 0.300 #0.300

#Scale the bounds of the target zone
LZ_scaled_mz1 = (LowerZone_mz1 - data_min_mz1[0])/(data_max_mz1[0] - data_min_mz1[0])
UZ_scaled_mz1 = (UpperZone_mz1 - data_min_mz1[0])/(data_max_mz1[0] - data_min_mz1[0])
LZ_scaled_mz2 = (LowerZone_mz2 - data_min_mz2[0])/(data_max_mz2[0] - data_min_mz2[0])
UZ_scaled_mz2 = (UpperZone_mz2 - data_min_mz2[0])/(data_max_mz2[0] - data_min_mz2[0])
LZ_scaled_mz3 = (LowerZone_mz3 - data_min_mz3[0])/(data_max_mz3[0] - data_min_mz3[0])
UZ_scaled_mz3 = (UpperZone_mz3 - data_min_mz3[0])/(data_max_mz3[0] - data_min_mz3[0])

#Set the bounds on the irrigation rate, the upper bounds are not same as that which were used to generate the LSTM models
# UL_mz1 = -(6.00*1e-07)*86400
# UB_mz1 = -(1.74*1e-07)*86400
# UL_mz2 = -(6.90*1e-07)*86400
# UB_mz2 = -(1.95*1e-07)*86400
# UL_mz3 = -(7.20*1e-07)*86400
# UB_mz3 = -(1.97*1e-07)*86400
# #Set the bounds on the irrigation rate, the upper bounds are not same as that which were used to generate the LSTM models
# UL_mz1 = -(6.00*1e-07)*86400
# UB_mz1 = -(1.97*1e-07)*86400
# UL_mz2 = -(6.90*1e-07)*86400
# UB_mz2 = -(1.97*1e-07)*86400
# UL_mz3 = -(7.20*1e-07)*86400
# UB_mz3 = -(1.97*1e-07)*86400
#Set the bounds on the irrigation rate, the upper bounds are not same as that which were used to generate the LSTM models
#Use these bounds when the rooting depth is 1.0 m 
# UL_mz1_mod = -(6.00*1e-07)*86400
# UB_mz1_mod = -(3.01*1e-07)*86400
# UL_mz2_mod = -(6.90*1e-07)*86400
# UB_mz2_mod = -(3.36*1e-07)*86400
# UL_mz3_mod = -(7.20*1e-07)*86400
# UB_mz3_mod = -(3.47*1e-07)*86400

# UL_mz1 = -(3.685*1e-07)*86400
# UB_mz1 = -(1.740*1e-07)*86400
# UL_mz2 = -(3.428*1e-07)*86400
# UB_mz2 = -(1.740*1e-07)*86400
# UL_mz3 = -(3.728*1e-07)*86400
# UB_mz3 = -(1.740*1e-07)*86400

# UL_mz1_mod = -(3.685*1e-07)*86400
# UB_mz1_mod = -(1.740*1e-07)*86400
# UL_mz2_mod = -(3.428*1e-07)*86400
# UB_mz2_mod = -(1.740*1e-07)*86400
# UL_mz3_mod = -(3.728*1e-07)*86400
# UB_mz3_mod = -(1.740*1e-07)*86400



UL_mz1 = -0.0318400
UB_mz1 = -0.0038880
UL_mz2 = -0.0296160
UB_mz2 = -0.0043200
UL_mz3 = -0.0322080
UB_mz3 = -0.0020112

UL_mz1_mod = -0.0318400
UB_mz1_mod = -0.0038880
UL_mz2_mod = -0.0296160
UB_mz2_mod = -0.0043200
UL_mz3_mod = -0.0322080
UB_mz3_mod = -0.0020112

# UL_mz1 = -(2.94*1e-07)*86400
# UB_mz1 = -(1.74*1e-07)*86400
# UL_mz2 = -(2.94*1e-07)*86400
# UB_mz2 = -(1.95*1e-07)*86400
# UL_mz3 = -(2.94*1e-07)*86400
# UB_mz3 = -(1.97*1e-07)*86400

# UL_mz1_mod = -(2.94*1e-07)*86400
# UB_mz1_mod = -(1.74*1e-07)*86400
# UL_mz2_mod = -(2.94*1e-07)*86400
# UB_mz2_mod = -(1.95*1e-07)*86400
# UL_mz3_mod = -(2.94*1e-07)*86400
# UB_mz3_mod = -(1.97*1e-07)*86400


# #Set the bounds on the irrigation rate, the upper bounds are not same as that which were used to generate the LSTM models
# #Use these bounds when the rooting depth is 1.0 m 
#UL_mz1_mod = -(6.00*1e-07)*86400
#UB_mz1_mod = -(2.89*1e-07)*86400
#UL_mz2_mod = -(6.90*1e-07)*86400
#UB_mz2_mod = -(2.70*1e-07)*86400
#UL_mz3_mod = -(7.20*1e-07)*86400
#UB_mz3_mod = -(2.60*1e-07)*86400

# #Set the bounds on the irrigation rate, the upper bounds are not same as that which were used to generate the LSTM models
# #Use these bounds when the rooting depth is 1.0 m 
# UL_mz1_mod = -(6.00*1e-07)*86400
# UB_mz1_mod = -(3.47*1e-07)*86400
# UL_mz2_mod = -(6.90*1e-07)*86400
# UB_mz2_mod = -(4.05*1e-07)*86400
# UL_mz3_mod = -(7.20*1e-07)*86400
# UB_mz3_mod = -(4.63*1e-07)*86400

#Penalty/Weighting matrices on the violation of the bounds of the target zone, and they can be tuned
# QUpper_mz1 = 2.0e07
# QLower_mz1 = 2.2e07
# QUpper_mz2 = 2.0e07
# QLower_mz2 = 2.2e07
# QUpper_mz3 = 2.0e07
# QLower_mz3 = 2.2e07

factor = 10000

QUpper_mz1 = factor*1.2e06
QLower_mz1 = factor*1.0e06
QUpper_mz2 = factor*1.2e06
QLower_mz2 = factor*1.0e06
QUpper_mz3 = factor*1.2e06
QLower_mz3 = factor*1.0e06

#This user defined function scales the original u amount for the management zones
def scaling_mz1(u):
    u = u/86400     
    return (u - data_min_mz1[1])/(data_max_mz1[1]-data_min_mz1[1])
def scaling_mz2(u):
    u = u/86400
    return (u - data_min_mz2[1])/(data_max_mz2[1]-data_min_mz2[1])
def scaling_mz3(u): 
    u = u/86400   
    return (u - data_min_mz3[1])/(data_max_mz3[1]-data_min_mz3[1])

def formulateOptimProb(ind, managementZone_index, currentStates, previousInputs, guessesX, guessesU, cropCoeff, refEvap, rooting_depths, lai_factors, rain, binaryValues, results):
    w = []
    J = 0
    lbw = []
    ubw = []
    Guess = []
    G = []
    lbg = []
    ubg = []
    pastStateValues = []
    pastInputValues = []
    
    for j in range(sequenceLength):
        xname = 'X' + str(j)
        Xk = cs.MX.sym(xname, 1)
        w+= [Xk]
        lbw+= [currentStates[j]]
        ubw+= [currentStates[j]]
        Guess+= [currentStates[j]]
        pastStateValues.append(Xk)
        pass
        
    for k in range(sequenceLength - 1):
        uname = 'U' +str(k)
        Uk = cs.MX.sym(uname, 1)
        w+= [Uk]
        lbw+= [previousInputs[k]]
        ubw+= [previousInputs[k]]
        Guess+= [previousInputs[k]]
        
        cname = 'C' + str(k)
        ck = cs.MX.sym(cname, 1)
        w+= [ck]
        lbw+=[1]
        ubw+=[1]
        Guess+=[1]
        pastInputValues.append(Uk)
        pass 


    for k in range(1, interPoints+1):
        # print(k)
        Uname = 'u' + str(k-1)
        Uk = cs.MX.sym(Uname, 1)    
        w+= [Uk]
        lbw+= [-np.inf]
        ubw+= [np.inf]
        Guess+= [guessesU[k-1]]
        pastInputValues.append(Uk)

        cname = 'C' + str(k-1) #Calculated by the RL agent
        Ck = cs.MX.sym(cname,1)
        w+=[Ck]
        lbw+=[binaryValues[k-1]]
        ubw+=[binaryValues[k-1]]
        Guess+=[binaryValues[k-1]]
        
        currentStateValues = pastStateValues[k-1 : (k-1) + sequenceLength]
        currentInputValues = pastInputValues[k-1 : (k-1) + sequenceLength]   
        currentCropCoefficient = cropCoeff[k-1 : (k-1) + sequenceLength]
        currentRefEvap = refEvap[k-1 : (k-1) + sequenceLength]
        currentRootingDepths = rooting_depths[k-1: (k-1) + sequenceLength]
        currentLAI_factors = lai_factors[k-1: (k-1) + sequenceLength]
        currentRainValues = rain[k-1 : (k-1) + sequenceLength] #Irrespective of the irrigation decision, the rain should have an effect on the system!

        currentPrecipValues_unscaled = []
        assert len(currentInputValues) == len(currentRainValues)
        for i in range(len(currentRainValues)):
            currentPrecipValues_unscaled.append(currentInputValues[i]+currentRainValues[i])
            pass 
        
        currentPrecipValues_scaled = []
        for i in range(len(currentRainValues)):
            if managementZone_index == 1:
                currentPrecipValues_scaled.append(scaling_mz1((currentPrecipValues_unscaled[i])))
            elif managementZone_index == 2:
                currentPrecipValues_scaled.append(scaling_mz2((currentPrecipValues_unscaled[i])))
            else:
                currentPrecipValues_scaled.append(scaling_mz3((currentPrecipValues_unscaled[i])))
            pass 

        #Determine the appropriate LSTM model based on the considered management zone
        if managementZone_index == 1:
            Fk = optimizeInputModel_automated_mz1(currentStateValues, currentPrecipValues_scaled, currentCropCoefficient, currentRefEvap, currentRootingDepths, currentLAI_factors)
        elif managementZone_index == 2:
            Fk = optimizeInputModel_automated_mz2(currentStateValues, currentPrecipValues_scaled, currentCropCoefficient, currentRefEvap, currentRootingDepths, currentLAI_factors)
        else: 
            Fk = optimizeInputModel_automated_mz3(currentStateValues, currentPrecipValues_scaled, currentCropCoefficient, currentRefEvap, currentRootingDepths, currentLAI_factors)
    
        Xname = 'X' +str(k)
        Xk=cs.MX.sym(Xname, 1)
        w+=[Xk]
        lbw+=[-np.inf]
        ubw+=[np.inf]
        Guess+=[guessesX[k-1]]
        pastStateValues.append(Xk)
        
        elName = 'el' + str(k)
        ekL = cs.MX.sym(elName, 1)  #The slack variable for the lower bound
        w+=[ekL]
        lbw+=[0]
        ubw+=[np.inf]
        Guess+=[0]
        
        euName = 'el'+ str(k)
        ekU = cs.MX.sym(euName, 1) #The slack variable for the upper bound
        w+=[ekU]
        lbw+=[0]
        ubw+=[np.inf]
        Guess+=[0]
            
        G+=[Fk - Xk]
        lbg+=[0]
        ubg+=[0]
        
        #Determine the bounds on the irrigation rate based on the considered management zone
        if managementZone_index  == 1:
            if rooting_depths[4+k-1] == 0:
                ubU = UB_mz1
                lbU = UL_mz1
            else:
                ubU = UB_mz1_mod
                lbU = UL_mz1_mod

        elif managementZone_index == 2:
            if rooting_depths[4+k-1] == 0:
                ubU = UB_mz2
                lbU = UL_mz2
            else:
                ubU = UB_mz2_mod
                lbU = UL_mz2_mod
        
        else:
            if rooting_depths[4+k-1] == 0:
                ubU = UB_mz3
                lbU = UL_mz3
            else:
                ubU = UB_mz3_mod
                lbU = UL_mz3_mod

        G+=[Uk - cs.mtimes(Ck, ubU)]
        lbg+=[-np.inf]
        ubg+=[0]
        G+=[Uk - cs.mtimes(Ck, lbU)]
        lbg+=[0]
        ubg+=[np.inf]
        
        #Determine the appropriate tuning parameters based on the considered management zone
        if managementZone_index == 1:
            QLower = QLower_mz1
            QUpper = QUpper_mz1
        elif managementZone_index == 2:
            QLower = QLower_mz2
            QUpper = QUpper_mz2
        else:
            QLower = QLower_mz3
            QUpper = QUpper_mz3

        #Cost function
        J+=cs.mtimes(cs.transpose(ekL), cs.mtimes(QLower, ekL)) + cs.mtimes(cs.transpose(ekU), cs.mtimes(QUpper, ekU)) - 9000*Uk

        # J+= cs.mtimes(QLower, cs.fabs(ekL)) +  cs.mtimes(QUpper, cs.fabs(ekU)) - 9000*Uk                                                                                                                               
    
        #Determine the bounds of the target zone based on the considered management zone	
        if managementZone_index == 1:
            lowerZone = LZ_scaled_mz1 
            upperZone = UZ_scaled_mz1
        elif managementZone_index == 2:
            lowerZone = LZ_scaled_mz2 
            upperZone = UZ_scaled_mz2
        else:
            lowerZone = LZ_scaled_mz3 
            upperZone = UZ_scaled_mz3
        G+=[Xk + ekL - ekU]
        lbg+=[lowerZone]
        ubg+=[upperZone]
        pass 

    #Solve the optimization problem, using IPOPT
    nlp=dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts = {"ipopt.print_level":1}
    S=cs.nlpsol('S','ipopt',nlp,opts)
    r=S(lbx=lbw, ubx=ubw,x0=Guess,lbg=lbg,ubg=ubg)
    optimalValues = r['x'].full().ravel()
    results.put((ind,optimalValues))
