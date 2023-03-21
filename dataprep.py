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
print(files)

# "file" in matlab is always last

import numpy as np
import h5py
f = h5py.File(f'./A2_labeling/responses/{files[2]}','r')
recordings = f.get('/')
recordings = np.array(recordings) # For converting to a NumPy array
print(recordings)
data = f.get(f"/{str(recordings[0])}/values")
data = np.array(data) # For converting to a NumPy array
print(data.shape)

