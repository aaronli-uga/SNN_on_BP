'''
Author: Qi7
Date: 2022-08-16 15:03:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-16 16:38:06
Description: 
'''
import numpy as np

'''
Classification of blood pressure for adults:
(0) Normal: SBP < 120 and DBP < 80
(1) Prehypertension: SBP 120-139 or DBP 80-89
(2) Stage 1: SBP 140-159 or DBP 90-99
(3) Stage 2: SBP >= 160 or DBP >= 100
'''

rawdata_train = "dataset/ctru_20_21_22_good_day_none_train.npy"
rawdata_test = "dataset/ctru_20_21_22_good_day_none_test.npy"
data_train = np.load(rawdata_train)
data_test = np.load(rawdata_test)

# we only look SBP in this case
N_train = np.where((data_train[:,-2] < 120) & (data_train[:,-1] < 80))
P_train = np.where((data_train[:,-2] <= 139) & (data_train[:,-2] >= 120))
S1_train = np.where((data_train[:,-2] <= 159) & (data_train[:,-2] >= 140))
S2_train = np.where((data_train[:,-2] >= 160))

N_test = np.where((data_test[:,-2] < 120) & (data_test[:,-1] < 80))
P_test = np.where((data_test[:,-2] <= 139) & (data_test[:,-2] >= 120))
S1_test = np.where((data_test[:,-2] <= 159) & (data_test[:,-2] >= 140))
S2_test = np.where((data_test[:,-2] >= 160))


N_data_train = data_train[N_train][:, 0:1000]
N_data_train = np.hstack((N_data_train, np.expand_dims(np.zeros(N_data_train.shape[0]), axis=1)))

P_data_train = data_train[P_train][:, 0:1000]
P_data_train = np.hstack((P_data_train, np.expand_dims(np.ones(P_data_train.shape[0]), axis=1)))

S1_data_train = data_train[S1_train][:, 0:1000]
S1_data_train = np.hstack((S1_data_train, np.expand_dims(np.ones(S1_data_train.shape[0]) * 2, axis=1)))

S2_data_train = data_train[S2_train][:, 0:1000]
S2_data_train = np.hstack((S2_data_train, np.expand_dims(np.ones(S2_data_train.shape[0]) * 3, axis=1)))


N_data_test = data_test[N_test][:, 0:1000]
N_data_test = np.hstack((N_data_test, np.expand_dims(np.zeros(N_data_test.shape[0]), axis=1)))

P_data_test = data_test[P_test][:, 0:1000]
P_data_test = np.hstack((P_data_test, np.expand_dims(np.ones(P_data_test.shape[0]), axis=1)))

S1_data_test = data_test[S1_test][:, 0:1000]
S1_data_test = np.hstack((S1_data_test, np.expand_dims(np.ones(S1_data_test.shape[0]) * 2, axis=1)))

S2_data_test = data_test[S2_test][:, 0:1000]
S2_data_test = np.hstack((S2_data_test, np.expand_dims(np.ones(S2_data_test.shape[0]) * 3, axis=1)))


npy_data_train = np.concatenate((N_data_train, P_data_train), axis=0)
npy_data_train = np.concatenate((npy_data_train, S1_data_train), axis=0)
npy_data_train = np.concatenate((npy_data_train, S2_data_train), axis=0)

npy_data_test = np.concatenate((N_data_test, P_data_test), axis=0)
npy_data_test = np.concatenate((npy_data_test, S1_data_test), axis=0)
npy_data_test = np.concatenate((npy_data_test, S2_data_test), axis=0)

# save the dataset, the last colomn is the label y.
with open('dataset/snn_train.npy', 'wb') as f:
    np.save(f, npy_data_train)

with open('dataset/snn_test.npy', 'wb') as f:
    np.save(f, npy_data_test)