import os
import numpy as np
import h5py

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
                if currAcc[i] != 0: 
                    output[i][currSec] = currAcc[i] / currSteps
                else:
                    output[i][currSec] = 0
                currAcc[i] = data[i][timesIndex]
            currSec += 1

        if times[0][timesIndex] > currSec + 2:
            for i in range(data.shape[0]):
                if currAcc[i] != 0: 
                    output[i][currSec] = currAcc[i] / currSteps
                    output[i][currSec + 1] = currAcc[i] / currSteps
                else:
                    output[i][currSec] = 0
                    output[i][currSec + 1] = 0
                currAcc[i] = data[i][timesIndex]
            currSec += 2
        timesIndex += 1
    return output
# get all filenames


# folder path
dir_path = r'.\A2_labeling\responses'

# list to store files
files = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)
print(files)

# "file" in matlab is always last

RECORDING_LENGTH = 3600
RESPONSES = 31

responsesToSave = np.zeros((RESPONSES,120,RECORDING_LENGTH))

count = 0
for file in files:
    f = h5py.File(f'./A2_labeling/responses/{file}','r')
    recordings = f.get('/')
    recordings = np.array(recordings) # For converting to a NumPy array

    for i in range(len(recordings)):
        if (recordings[i] != "file"):
            times = f.get(f"/{str(recordings[i])}/times")
            times = np.array(times) # For converting to a NumPy array
            print(times.shape)


            data = f.get(f"/{str(recordings[i])}/values")
            data = np.array(data) # For converting to a NumPy array
            print(data.shape)

            # save
            responsesToSave[count] = binToSec(data,times, RECORDING_LENGTH)
            count += 1


# print(binToSec(data, times, 3600))
# print(binToSec(data, times, 3600).shape)

print(responsesToSave)
print(responsesToSave.shape)

np.save("responses.npy", responsesToSave)