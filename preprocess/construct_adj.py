import pickle as pkl
import numpy as np
import scipy.sparse as ss

# with open('cluster/data_afcluster/Trace_array', 'rb') as f:
#     Trace_array = pkl.load(f)

# Trace_array = np.load('../privacy/no_noise/trace_array.npy', allow_pickle=True)
un = 10
rz = True
H = np.load("../privacy/noposterior/H_un={}_rm01={}.npy".format(un, rz))

Meet_frequency = np.dot(H, H.T)
Adj = np.where(Meet_frequency == 0, 0, 1)
np.save('../privacy/noposterior/A_un={}_r01={}.npy'.format(un, rz), Adj)
# # get the meet_frequency matrix
# def construct_network(trace_array):
#     pop_num = trace_array.shape[0]
#     meet_frequency = np.zeros((pop_num, pop_num))
#
#     for i in range(pop_num):
#         for j in range(i, pop_num):
#             temp = trace_array[i] - trace_array[j]
#             meet_frequency[i, j] = np.sum(np.where(temp == 0, 1, 0))
#             meet_frequency[j, i] = meet_frequency[i, j]
#     print('Construct network successfully!')
#     return meet_frequency
#
# user_num = Trace_array.shape[0]
# slot_num = Trace_array.shape[1]
#
# Meet_frequency = construct_network(Trace_array)
# Adj = np.where(Meet_frequency == 0, 0, 1)
#
# with open('cluster/data_afcluster/Adj', 'wb') as f:
#     pkl.dump(Adj, f)
