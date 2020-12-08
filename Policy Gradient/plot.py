import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

    
def generate_training_plot(file_1, file_2: str):
    default_rewards = []
    engine_noise_rewards = []

    # open file and read the content in a list
    with open(file_1, 'r') as filehandle:
        for i, line in enumerate(filehandle):
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            default_rewards.append([float(currentPlace), i+1])
    
    with open(file_2, 'r') as filehandle:
        for i, line in enumerate(filehandle):
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            engine_noise_rewards.append([float(currentPlace), i+1])


    df_default_rewards = pd.DataFrame(default_rewards, columns=['Reward', 'Episode'])
    df_engine_noise_rewards = pd.DataFrame(engine_noise_rewards, columns=['Reward', 'Episode'])

    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Learning Performance')
    plt.plot(df_engine_noise_rewards['Episode'], df_engine_noise_rewards['Reward'], '-o', label='Sensor Noise ~ std: 0.5')
    plt.plot(df_default_rewards['Episode'], df_default_rewards['Reward'], '-o', label='Default')
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig(f"metrics/sensor_high_comparison.png", format='png', dpi=400, bbox_inches='tight')


if __name__ == "__main__":
     generate_training_plot(sys.argv[1], sys.argv[2])
