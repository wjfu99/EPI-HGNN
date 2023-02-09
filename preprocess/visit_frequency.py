import numpy as np
from collections import Counter
from tqdm import tqdm
import scipy.sparse as ss

region_num = 661
region_num = 11459
user_num = 15279

trace_array = np.load('../privacy/noposterior/trace_array.npy')

# for i in trace_array.shape[0]:
#     temp = np.bincount(trace_array[i, :])
# time_slot = trace_array.shape[1]


vs_time_matrix = ss.dok_matrix((user_num, region_num))

vs_time_matrix = np.zeros((user_num, region_num), dtype=np.int32)

for user in tqdm(range(trace_array.shape[0])):
    temp = Counter(trace_array[user, :])
    del temp[-1]
    for region in temp:
        vs_time_matrix[user, region] = temp[region]


np.save("../privacy/noposterior/visit_frequency.npy", vs_time_matrix)
