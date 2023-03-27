import os
import numpy as np
import h5py

def binToSec(data, length, bin):
    timesIndex = 0
    currAcc = np.zeros((int(length/bin), 1))
    # for i in range(int(length/bin)):
    for i in range(currAcc.shape[0]):
        inBin = 0
        if timesIndex >= data.shape[1]:
            break  
        while (data[0][timesIndex] < (i + 1) * bin):
            inBin += 1
            timesIndex += 1
            if timesIndex >= data.shape[1]:
                break  
            # print(timesIndex)
        currAcc[i] = inBin
    return currAcc 

# get all filenames


# folder path
dir_path = r'.\A2_labeling\non_responsive'

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
BINNING = 5
RESPONSES = 190

responsesToSave = np.zeros((RESPONSES,int(RECORDING_LENGTH/5),1))

count = 0
for file in files:
    f = h5py.File(f'./A2_labeling/non_responsive/{file}','r')
    recordings = f.get('/')
    recordings = np.array(recordings) # For converting to a NumPy array

    for i in range(len(recordings)):
        if (recordings[i] != "file"):
            times = f.get(f"/{str(recordings[i])}/times")
            times = np.array(times) # For converting to a NumPy array
            print(times.shape)
            # save
            responsesToSave[count] = binToSec(times, RECORDING_LENGTH, BINNING)
            count += 1
            print(f"Done {count} of {RESPONSES}")


# print(binToSec(data, times, 3600))
# print(binToSec(data, times, 3600).shape)

print(responsesToSave)
print(responsesToSave.shape)

# np.save("non-responses.npy", responsesToSave)