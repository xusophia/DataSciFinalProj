import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


training_record = []

# open file and read the content in a list
with open('rewards_log.txt', 'r') as filehandle:
    for i, line in enumerate(filehandle):
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        training_record.append([float(currentPlace), i+1])

df_training_record = pd.DataFrame(training_record, columns=['Reward', 'Episode'])

fig = plt.figure()
plt.xlabel('Episode')
plt.ylabel('Learning Performance')
plt.plot(df_training_record['Episode'], df_training_record['Reward'], '-o')
plt.show()
fig.savefig('training_record.jpeg', format='jpeg', dpi=400, bbox_inches='tight')
