import numpy as np
import datetime as db
from datetime import datetime as da
from tqdm import tqdm
import copy

def count_first_and_last(user_traj):
    first=[]
    last=[]
    for i in user_traj:
        key_list = list(user_traj[i].keys())
        first.append(key_list[0])
        last.append(key_list[-1])
    return first, last


def get_max(arr):
    last_set=set(arr)
    uu1=[]
    #from tqdm import tqdm
    for i in tqdm(last_set):
        uu1.append((i,arr.count(i)))
    max_last=sorted(uu1,key=lambda x:x[1])
    return max_last[-1][0]


def map_user_dict(start_time, end_time, user_traj):
    user_dict = {}
    for u in tqdm(user_traj):
        num = (end_time - start_time).total_seconds() // 1800 + 1
        user_dict[u] = {}
        for n in range(int(num)):
            tim = start_time + n * db.timedelta(minutes=30)
            user_dict[u][tim] = -1
        for j in user_traj[u]:
            if j <= end_time and j >= start_time:
                user_dict[u][j] = user_traj[u][j]
    return user_dict

def id_map(user_dict):
    user_id = {}
    all_dict = {}
    for i, j in enumerate(user_dict):
        all_dict[i] = user_dict[j]
        user_id[i] = j
    np.save('user_id.npy', user_id)
    return user_id, all_dict


def traj_count_poi(user_traj, all_dict):
    locations = set()
    for i in tqdm(user_traj):
        traj = list(user_traj[i].values())
        for j in traj:
            locations.add(j)
    loc_list = list(locations)
    trace_array = []
    for info in all_dict:
        trace_array.append(len(all_dict[info]))
    poi_id = {}
    for i in range(len(loc_list)):
        poi_id[loc_list[i]] = i
    return locations, trace_array, poi_id

def data_process_save(all_dict, poi_id):
    a1_d = {}
    for u in tqdm(all_dict):
        a1_d[u] = {}
        for j in all_dict[u]:
            if all_dict[u][j] != -1:
                a1_d[u][j] = poi_id[all_dict[u][j]]
            else:
                a1_d[u][j] = -1
    a2d = {}
    for i in a1_d:
        a2d[i] = {}
        a2d[i]['trace'] = list(a1_d[i].values())
    np.save('ori_data_old.npy', a2d)

def main():

    user_traj = np.load('final_user.npy', allow_pickle=True).item()
    first, last = count_first_and_last(user_traj)

    start_time = get_max(first)
    end_time = get_max(last)

    user_dict = map_user_dict(start_time, end_time, user_traj)

    user_id, all_dict = id_map(user_dict)

    locations, trace_array, poi_id = traj_count_poi(user_traj, all_dict)

    # poi_id = np.load("region_cluster/region_id.npy", allow_pickle=True).item()
    data_process_save(all_dict, poi_id)


if __name__ == "__main__":
    main()







