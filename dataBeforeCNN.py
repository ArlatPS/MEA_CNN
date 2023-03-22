import numpy as np

responses = np.load("responses.npy")
non_responses = np.load("non-responses.npy")



# print(zeros)
# print(ones)

# take 15% + 1 of data for testing randomly
# zeros
## more balanced validation data - bigger to be later taken
N_zeros = 190
N_zeros_to_test = 35
zeros_tst = np.zeros((N_zeros_to_test, 120, 3600))
ZT_count = -1;
new_zeros = np.zeros((N_zeros - N_zeros_to_test, 120,3600))
NZ_count = -1
idx = np.argsort(np.random.random(N_zeros))
for i in range(N_zeros):
    if i in idx[0:N_zeros_to_test]: 
        ZT_count += 1
        zeros_tst[ZT_count] = non_responses[i]
    else: 
        NZ_count +=1
        new_zeros[NZ_count] = non_responses[i]

N_ones = 31;
N_ones_to_test = 14;
ones_tst = np.zeros((N_ones_to_test, 120,3600))
ZT_count = -1;
new_ones = np.zeros((N_ones - N_ones_to_test, 120,3600))
NZ_count = -1
idx = np.argsort(np.random.random(N_ones))
for i in range(N_ones):
    if i in idx[0:N_ones_to_test]: 
        ZT_count += 1
        ones_tst[ZT_count] = responses[i]
    else: 
        NZ_count +=1
        new_ones[NZ_count] = responses[i]



# data for learning shuffled 
data = np.concatenate((new_zeros, new_ones))
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


np.save("x_trn.npy", data)
np.save("y_trn.npy", labels)

np.save("non-responses_tst.npy", zeros_tst)
np.save("responses_tst.npy", ones_tst)