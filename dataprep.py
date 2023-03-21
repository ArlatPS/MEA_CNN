# get all filenames

import os

# folder path
dir_path = r'.\A2_labeling\responses'

# list to store files
files = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)
# print(files)

# "file" in matlab is always last

RECORDING_LENGTH = 3600

import numpy as np
import h5py
f = h5py.File(f'./A2_labeling/responses/{files[0]}','r')
recordings = f.get('/')
recordings = np.array(recordings) # For converting to a NumPy array
print(recordings)

times = f.get(f"/{str(recordings[0])}/times")
times = np.array(times) # For converting to a NumPy array
print(times.shape)


data = f.get(f"/{str(recordings[0])}/values")
data = np.array(data) # For converting to a NumPy array
print(data.shape)

def binToSec(data, times, length):
    currSec = 0
    timesIndex = 0
    currAcc = np.zeros((data.shape[0], 1))
    currSteps = 0
    output = np.zeros((data.shape[0], length))
    while (timesIndex < times.shape[1]):
        if times[0][timesIndex] < currSec + 1:
            currSteps += 1
            for i in range(data.shape[0]):
                currAcc[i] += data[i][timesIndex]

        if times[0][timesIndex] > currSec + 1 and times[0][timesIndex] < currSec + 2:
            for i in range(data.shape[0]):
                output[i][currSec] = currAcc[i] / currSteps
                currAcc[i] = data[i][timesIndex]
            currSec += 1
            
        if times[0][timesIndex] > currSec + 2:
            for i in range(data.shape[0]):
                output[i][currSec] = currAcc[i] / currSteps
                output[i][currSec + 1] = currAcc[i] / currSteps
                currAcc[i] = data[i][timesIndex]
            currSec += 2
        timesIndex += 1
    return output

print(binToSec(data, times, 3600))
print(binToSec(data, times, 3600).shape)
