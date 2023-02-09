from enum import Flag

import numpy as np
from tqdm import tqdm
import pickle as pkl
import scipy.sparse as ss
from scipy.spatial import KDTree
from collections import Counter
import random

# f = np.load('final_result/ori_data.npy', allow_pickle=True).item()
# graph = np.load('final_result/graph_dict.npy', allow_pickle=True).item()
# sample_result = np.load('final_result/sample_result.npy', allow_pickle=True).item()


per_data = np.load('../region_cluster/per_data_old.npy', allow_pickle=True).item()
# per_data1 = np.load('../ori_data_old.npy', allow_pickle=True).item()

eps = 400

m = 10
Pop_num = 15279

# Region_num = 661

Region_num = 11459
un = 10
noise_data = True
obfuscated_unique = True

freq = False
truncation = False
rz = True

posterior = True
if obfuscated_unique:
    ori_data = np.load('../obfuscate_data/ori_data_eps={}.npy'.format(eps), allow_pickle=True).item()
else:
    ori_data = np.load('../obfuscate_data_old/ori_data_eps={}.npy'.format(eps), allow_pickle=True).item()

if noise_data:
    pop_info = ori_data
else:
    pop_info = per_data[0]
loc_id_dict = np.load('../loc_id_dict.npy', allow_pickle=True).item()
poi_id = np.load('../poi_id.npy', allow_pickle=True).item()
poi_id = dict(zip(poi_id.values(), poi_id.keys()))
poi_loc = np.load('../poi_loc.npy', allow_pickle=True).item()

# %% preprocess KDTree
spatial_data = np.array(list(loc_id_dict.keys()))
kd_tree = KDTree(data=spatial_data)

def dit2array(trace_dit):
    trace_array = np.full((Pop_num, 48*40), -1, dtype=np.int)
    for i, key in enumerate(trace_dit):
        trace_array[i, :] = trace_dit[key]['trace'][:48*40]
    return trace_array

trace_array = dit2array(pop_info)


# %%
def construct_hyper_network(trace_array, unique_num=1, rm_zero=True):
    trace = np.split(trace_array, unique_num, axis=1)
    slot_num = trace_array.shape[1]
    if slot_num % unique_num != 0:
        print("choose a appropriate unique_num!")
        exit()
    unit_len = slot_num//unique_num
    h = np.zeros([Pop_num, Region_num*unique_num])
    t = np.zeros((Region_num*unique_num, 2))

    for day in tqdm(range(unique_num)):

        
        temp_trace = trace[day]
        (pop_num, time_num) = temp_trace.shape
        
        t[day*Region_num: (day+1)*Region_num, :] = [day*unit_len, (day+1)*unit_len - 1]
        # t[day*Region_num:(day+1)*Region_num, day*(time_num*Embedding_dim):(day+1)*(time_num*Embedding_dim)] = 1
        for i in range(pop_num):
            region_sets = set(temp_trace[i])
            region_num = Counter(temp_trace[i])
            n = temp_trace.shape[1] - region_num[-1]
            for region in region_sets:
                
                if region >= 0:
                    if posterior:
                        # loc = loc_id_dict[poi_loc[str[region]]]
                        loc = poi_loc[str(poi_id[region])]
                        distance, nearest_m_id = kd_tree.query(loc, k=m)
                        c = np.exp(-eps*distance)
                       
                        nearest_m_pts = kd_tree.data[nearest_m_id]
                        
                        poi_num, poi_lists = [], []
                        for pts in nearest_m_pts:
                            poi_lists.append(loc_id_dict[tuple(pts)])
                            poi_num.append(len(loc_id_dict[tuple(pts)]))
                        c = c * poi_num
                        c = c / c.sum()
                        for c_idx, poi_list in enumerate(poi_lists):
                            for poi in poi_list:
                                
                                if freq:
                                    h[i][day * Region_num + poi] += c[c_idx] / (len(poi_list) * n) * region_num[region]
                                else:
                                    h[i][day*Region_num + poi] += c[c_idx] / (len(poi_list) * (len(region_sets) - 1))
                    else:
                        # h[i][day * Region_num + region] = 1
                        h[i][day*Region_num + region] = 1 / (len(region_sets) - 1)
            if truncation:
                w = h[i]
                # idx = w > 0
                idx = list(np.argwhere(w > 0).flatten())
                w = w[idx]
                # t = tuple(zip(idx, w))
                # t = sorted(t, key=lambda tup: tup[1], reverse=True)
                # t = t[:5]
                # idx, w = list(zip(*t))
                h[i] = 0
                h[i][idx] = w
            
            # w = h[i]
            # idx = w > 0
            # idx1 = list(np.argwhere(idx == True).flatten())
            # w = w[idx]
            # t = tuple(zip(idx1, w))
            # t = sorted(t, key=lambda tup: tup[1])
    if rm_zero:
        # temp = np.any(h, axis=0)
        temp = np.count_nonzero(h, axis=0)
        
        temp = np.where(temp > 1, True, False)
        h = h[:, temp]
        t = t[temp, :]
    return h, t


H, T = construct_hyper_network(trace_array, unique_num=un, rm_zero=rz)
# H = ss.coo_matrix(H)
np.save("../privacy/noposterior/trace_array.npy", trace_array)
np.save("../privacy/noposterior/H_un={}_rm01={}.npy".format(un, rz), H)
np.save("../privacy/noposterior/T_un={}_rm01={}.npy".format(un, rz), T)
# with open("H_un={}_r01={}".format(un, rz), 'wb') as f:
#     pkl.dump(H, f)
#
# with open("final_result/T_un={}_r01={}".format(un, rz), 'wb') as f:
#     pkl.dump(T, f)