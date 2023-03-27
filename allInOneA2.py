import os
import numpy as np
import h5py
from sklearn import decomposition
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
 
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
sum_zeros = 0
sum_ones = 0
reps = 20

for i in range(reps):

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
    BINNING = 10
    LENGTH = int(RECORDING_LENGTH / BINNING)
    RESPONSES = 190

    non_responsive = np.zeros((RESPONSES,int(RECORDING_LENGTH/BINNING),1))

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
                non_responsive[count] = binToSec(times, RECORDING_LENGTH, BINNING)
                count += 1
                print(f"Done {count} of {RESPONSES}")


    dir_path = r'.\A2_labeling\responses'

    # list to store files
    files = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            files.append(path)

    responsive = np.zeros((RESPONSES,int(RECORDING_LENGTH/BINNING),1))
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
                # save
                responsive[count] = binToSec(times, RECORDING_LENGTH, BINNING)
                count += 1
                print(f"Done {count} of {RESPONSES}")


    N_zeros = 190
    N_zeros_to_test = 30;
    zeros_tst = np.zeros((N_zeros_to_test, LENGTH))
    ZT_count = -1;
    new_zeros = np.zeros((N_zeros - N_zeros_to_test, LENGTH))
    NZ_count = -1
    idx = np.argsort(np.random.random(N_zeros))
    for i in range(N_zeros):
        if i in idx[0:N_zeros_to_test]: 
            ZT_count += 1
            zeros_tst[ZT_count] = non_responsive[i][0]
        else: 
            NZ_count +=1
            new_zeros[NZ_count] = non_responsive[i][0]

    N_ones = 31;
    N_ones_to_test = 8;
    ones_tst = np.zeros((N_ones_to_test, LENGTH))
    ZT_count = -1;
    new_ones = np.zeros((N_ones - N_ones_to_test, LENGTH))
    NZ_count = -1
    idx = np.argsort(np.random.random(N_ones))
    for i in range(N_ones):
        if i in idx[0:N_ones_to_test]: 
            ZT_count += 1
            ones_tst[ZT_count] = responsive[i][0]
        else: 
            NZ_count +=1
            new_ones[NZ_count] = responsive[i][0]

    def generateData(pca, x, start):
        original = pca.components_.copy()
        ncomp = pca.components_.shape[0]
        a = pca.transform(x)
        for i in range(start, ncomp):
            pca.components_[i,:] += np.random.normal(scale=0.1, size=ncomp)
            b = pca.inverse_transform(a)
            pca.components_ = original.copy()
            return b

    print(new_ones.shape)
    pca = decomposition.PCA(n_components=360)
    pca.fit(np.concatenate((new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,new_ones, new_ones,)))

    start = 3
    nsets = 10
    nsamp = new_ones.shape[0]
    new_unos = np.zeros((nsets*nsamp, new_ones.shape[1]))

    for i in range(nsets):
        if (i == 0):
            new_unos[0:nsamp,:] = new_ones
        else:
            new_unos[(i * nsamp):((i + 1) * nsamp), :] = generateData(pca,new_ones,start)


    # data for learning shuffled 
    data = np.concatenate((new_zeros, new_unos))
    labels = np.concatenate((np.zeros(N_zeros - N_zeros_to_test), np.ones(N_ones - N_ones_to_test)))

    # new indexes
    idx = np.argsort(np.random.random(N_zeros - N_zeros_to_test + N_ones - N_ones_to_test ));
    data = data[idx]
    labels = labels[idx]




    # # testing data
    # data_tst = np.concatenate((zeros_tst, ones_tst))
    # labels_tst = np.concatenate((np.zeros(20), np.ones(4)))

    print(data.shape)
    print(labels.shape)
    print(zeros_tst.shape)
    print(ones_tst.shape)







    x_trn = data
    y_trn = labels

    pca = decomposition.PCA(n_components=360)
    pca.fit(np.concatenate((x_trn, x_trn)))
    print(pca.explained_variance_ratio_)
    print(x_trn.shape)

    start = 3
    nsets = 20
    nsamp = x_trn.shape[0]
    new_x = np.zeros((nsets*nsamp, x_trn.shape[1]))
    new_y = np.zeros((nsets*nsamp))

    for i in range(nsets):
        if (i == 0):
            new_x[0:nsamp,:] = x_trn
            new_y[0:nsamp] = y_trn
        else:
            new_x[(i * nsamp):((i + 1) * nsamp), :] = generateData(pca,x_trn,start)
            new_y[(i*nsamp):((i + 1)*nsamp)] = y_trn

    idx = np.argsort(np.random.random(nsets * nsamp))
    new_x = new_x[idx]
    new_y = new_y[idx]



    # load data
    x_trn = new_x
    y_trn = new_y
    # x_test = np.load('x_tst.npy')
    # y_test = np.load('y_tst.npy')


    # adding 10% validation data from training (its shuffled)
    N = 120

    x_val = x_trn[:N]
    x_trn = x_trn[N:]
    y_val = y_trn[:N]
    y_trn = y_trn[N:]

    batch_size = 8
    epochs = 50

    model = Sequential()
    model.add(Input(shape=(LENGTH,)))
    model.add(Dense(512, activation="relu", bias_initializer="glorot_uniform"))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation="relu", bias_initializer="glorot_uniform"))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.045),loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    print('Model parameters = %d' % model.count_params())
    print(model.summary())

    history = model.fit(x_trn, y_trn, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val,y_val))

    score = model.evaluate(zeros_tst ,np.zeros(zeros_tst.shape[0]), verbose=0)
    score2 = model.evaluate(ones_tst ,np.ones(ones_tst.shape[0]), verbose=0)


    print('Test loss:     ', score[0])
    print('Test accuracy: ', score[1])
    print('Test loss:     ', score2[0])
    print('Test accuracy: ', score2[1])
    sum_zeros += score[1]
    sum_ones += score2[1]

print(f"Average accuracy for non-responsive: {sum_zeros / reps}")
print(f"Average accuracy for responsive: {sum_ones / reps}")
print(f"Global accuracy: {(sum_ones / reps + sum_zeros / reps)/2}")