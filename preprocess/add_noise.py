import numpy as np
from laplacian_noise import planar_laplacian
import copy
from scipy.spatial import KDTree
from tqdm import tqdm

# True for one poi in one user's trace will be obfuscated to same poi.
obfuscated_unique = True
obfuscated_itself = True

# %%
# from region_cluster.test import ori_data
# ori_data = np.load('ori_data.npy', allow_pickle=True).item()
ori_data_old = np.load('ori_data_old.npy', allow_pickle=True).item()
# %%

poi_id = np.load('poi_id.npy',allow_pickle=True).item()
poi_id = dict(zip(poi_id.values(), poi_id.keys()))
poi_loc = np.load('poi_loc.npy', allow_pickle=True).item()

noise_data = copy.deepcopy(ori_data_old)

# %% preprocess KDTree
spatial_data = np.array([poi_loc[str(uid)] for uid in poi_id.values()])
kd_tree = KDTree(data=spatial_data)

# %% preprocess loc_id_dict
loc_id_dict = {}
for id, uid in tqdm(poi_id.items()):
    if poi_loc[str(uid)] not in loc_id_dict:
        loc_id_dict[poi_loc[str(uid)]] = [id]
    # if this loc is occured
    else:
        loc_id_dict[poi_loc[str(uid)]].append(id)
        # print(poi_id[id], poi_id[loc_id_dict[poi_loc[str(uid)]]])
obfuscated_dict = {}
# %% add noise
for eps in tqdm([int(_*100) for _ in [0.5]]):
    for usr_id, usr in tqdm(ori_data_old.items(), leave=False):
        if obfuscated_unique:
            poi_set = set(usr['trace'])
            
            noise_data[usr_id]['trace'] = np.array(usr['trace'])
            for poi in poi_set:
                if poi >= 0:
                    id = poi_id[poi]
                    loc = poi_loc[str(id)]
                    obfuscate_loc, confidence = planar_laplacian(loc, loc_id_dict, kd_tree, eps=eps)
                    if not obfuscated_itself:
                        while obfuscate_loc == poi:
                            obfuscate_loc, confidence = planar_laplacian(loc, loc_id_dict, kd_tree, eps=eps)
                        assert obfuscate_loc != poi_loc
                    # trace = usr['trace']
                    trace = noise_data[usr_id]['trace']
                    
                    noise_data[usr_id]['trace'] = np.where(np.array(usr['trace']) == poi, obfuscate_loc, trace)
                    # idxs = np.arrry(usr['trace']) == poi
                    # noise_data[usr_id]['trace'][idxs] = obfuscate_loc
        else:
            for idx, poi in enumerate(usr['trace']):
                if poi != -1:
                    id = poi_id[poi]
                    loc = poi_loc[str(id)]
                    obfuscate_loc, confidence = planar_laplacian(loc, loc_id_dict, kd_tree, eps=eps)
                    noise_data[usr_id]['trace'][idx] = obfuscate_loc
    if obfuscated_unique:
        np.save('./obfuscate_data/ori_data_eps={}'.format(eps), noise_data)
    else:
        np.save('./obfuscate_data_old/ori_data_eps={}'.format(eps), noise_data)
