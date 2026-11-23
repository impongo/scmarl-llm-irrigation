from matplotlib import pyplot as plt
import numpy as np  

def GenerateTrajectories(episode_length, prescribed_irrigation, pressure_head,lower_zone, upper_zone, filename):
    time_states = np.arange(episode_length+1) +1
    time_inputs = np.arange(episode_length)+1

    fig, axs = plt.subplots(2,1, figsize=(8,8))
    axs[0].plot(time_inputs, prescribed_irrigation*100, marker='.', linewidth=2)
    axs[0].set_title("Presecribed irrigation rate")
    axs[0].set_xlim(xmin=1)
    #axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Irrigation rate (cm/day)")

    axs[1].plot(time_states, lower_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, upper_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, pressure_head, color='red')
    #axs[1].legend(['state Trajectory', 'zone'])
    axs[1].set_xlim(xmin=1)
    axs[1].set_title('Root zone capillary pressure head')
    axs[1].set_ylabel('Pressure Head (m)')
    axs[1].set_xlabel('Time (days)')
    fig.tight_layout()
    plt.savefig("./Current_results/"+filename+".pdf")
    return 

def GenerateTrajectories_OL(episode_length, prescribed_irrigation, pressure_head,lower_zone, upper_zone):
    time_states = np.arange(episode_length+1) +1
    time_inputs = np.arange(episode_length)+1
    fig, axs = plt.subplots(2,1, figsize=(8,8))
    axs[0].plot(time_inputs, prescribed_irrigation*100, marker='.', linewidth=2)
    axs[0].set_title("Presecribed irrigation rate")
    axs[0].set_xlim(xmin=1)
    #axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Irrigation rate (cm/day)")
    axs[1].plot(time_states, lower_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, upper_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, pressure_head, color='red')
    #axs[1].legend(['state Trajectory', 'zone'])
    axs[1].set_xlim(xmin=1)
    axs[1].set_title('Root zone capillary pressure head')
    axs[1].set_ylabel('Pressure Head (m)')
    axs[1].set_xlabel('Time (days)')
    fig.tight_layout()
    # Batch simulations should never open interactive windows.
    plt.close(fig)
    return 
    

def PlotRewardTrajectory(rewardTrajectory, filename):
    n_episodes = len(rewardTrajectory)
    n_episodes_array = np.arange(n_episodes) + 1 
    plt.figure(2, figsize=(9,9))
    plt.plot(n_episodes_array,rewardTrajectory)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("./Current_results/"+filename+".pdf")
    return
