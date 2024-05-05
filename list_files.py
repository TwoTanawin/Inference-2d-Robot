import os

folder_path = '/home/two-intel/Documents/Two/ait/reinforcement-learning/2d-robot-arm-2DoF-DRL/stb3/report/A2C_MLP_Robot2DoF/model'
files = os.listdir(folder_path)

for file in files:
    print(folder_path+file)
