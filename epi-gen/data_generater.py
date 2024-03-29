import numpy as np
import random
from tqdm import tqdm

def init_infect(pop_info,ratio):
    for info in pop_info:
        if random.uniform(0, 1) < ratio:
            pop_info[info]['state'] = 'I'
        else:
            pop_info[info]['state'] = 'S'
    return pop_info


def gleam_epidemic(pop_info,period,ori,end):
    trace_array = []
    for info in pop_info:
        trace_array.append(pop_info[info]['trace'][ori*period*48:48*period*end])
    trace_array = np.array(trace_array)
    print(trace_array.shape)
    for j in tqdm(range(trace_array.shape[1])):
        region_infected_num = np.zeros(Region_num, dtype=int)
        region_pop_num = np.zeros(Region_num, dtype=int)
        for i in pop_info:
            if pop_info[i]['trace'][j]!=-1:
                if pop_info[i]['state'] == 'I':
                    region_infected_num[pop_info[i]['trace'][j]] += 1
                region_pop_num[pop_info[i]['trace'][j]] += 1
        region_pop_num += np.where(region_pop_num == 0, 1, 0)
        region_force = Beta * np.true_divide(region_infected_num, region_pop_num)
        for info in pop_info:
            if pop_info[info]['trace'][j]!=-1:
                if pop_info[info]['state'] == 'S':
                    if random.uniform(0, 1) < region_force[pop_info[info]['trace'][j]]:
                        pop_info[info]['state'] = 'I'
                elif pop_info[info]['state'] == 'I':
                    if random.uniform(0, 1) < Mu:
                        pop_info[info]['state'] = 'R'
    return pop_info


def on_gleam_epidemic(pop_info, ori, period):
    trace_array = []
    for info in pop_info:
        trace_array.append(pop_info[info]['trace'][ori*period*48:])
    trace_array = np.array(trace_array)
    print(trace_array.shape)
    for j in tqdm(range(trace_array.shape[1])):
        region_infected_num = np.zeros(Region_num, dtype=int)
        region_pop_num = np.zeros(Region_num, dtype=int)
        for i in pop_info:
            if pop_info[i]['trace'][j]!=-1:
                if pop_info[i]['state'] == 'I':
                    region_infected_num[pop_info[i]['trace'][j]] += 1
                region_pop_num[pop_info[i]['trace'][j]] += 1
        region_pop_num += np.where(region_pop_num == 0, 1, 0)
        region_force = Beta * np.true_divide(region_infected_num, region_pop_num)
        for info in pop_info:
            if pop_info[info]['trace'][j]!=-1:
                if pop_info[info]['state'] == 'S':
                    if random.uniform(0, 1) < region_force[pop_info[info]['trace'][j]]:
                        pop_info[info]['state'] = 'I'
                elif pop_info[info]['state'] == 'I':
                    if random.uniform(0, 1) < Mu:
                        pop_info[info]['state'] = 'R'
    return pop_info


def isolate(pop_info,num,period):
    sample_dict = {}
    new_dict = {}
    iso = {}
    period_data={}
    for i in range(num):
        sample_dict[i]={}
        iso[i] = {}
        new_info = {}
        period_data[i]={}
        pop_info = gleam_epidemic(pop_info,period,i,i+1)
        for p in pop_info:
            period_data[i][p]={}
            if pop_info[p]['state']=='S':
                period_data[i][p]['state']=0
            elif pop_info[p]['state'] == 'I':
                period_data[i][p]['state'] = 1
            else:
                period_data[i][p]['state'] = 2
            period_data[i][p]['trace']=pop_info[p]['trace'][i*period*48:48*period*(i+1)]
        sample_result = random.sample(list(pop_info.keys()),int(len(pop_info)*0.1))
        for j in sample_result:
            sample_dict[i][j]=pop_info[j]
        for j in sample_result:
            if pop_info[j]['state']!='S':
                iso[i][j]=pop_info[j]

        for j in pop_info:
            if j not in sample_result:
                new_info[j]=pop_info[j]
            elif j in sample_result and pop_info[j]['state']=='S':
                new_info[j]=pop_info[j]
        pop_info = new_info
    pop_info = on_gleam_epidemic(pop_info,period,4)
    period_data[4]={}
    for p in pop_info:
        period_data[4][p]={}
        if pop_info[p]['state']=='S':
            period_data[4][p]['state']=0
        elif pop_info[p]['state']=='I':
            period_data[4][p]['state']=1
        else:
            period_data[4][p]['state']=2
        period_data[4][p]['trace']=pop_info[p]['trace'][4*10*48:]
    return sample_dict,pop_info,iso,period_data

def main():
    random.seed(123)
    global Region_num
    global Mu
    global Beta

    Region_num = 11459
    parameters = 'Omicron'
    if parameters == 'Omicron':
        # Omicron
        Mu = 0.0030639024815920513
        Beta = 0.058656271323618725
    elif parameters == 'primitive':
        # primitive
        Mu = 0.0029745672532136558
        Beta = 0.019132277144042642
    ratio = 0.02
    user_info = np.load('ori_data_old.npy', allow_pickle=True).item()
    # for i in range(44000):
    #     user_info.pop(random.choice(user_info.keys()))
    # for key in random.sample(user_info.keys(), 44000):
    #     del user_info[key]
    pop_info = init_infect(user_info,ratio)
    sample_dict,pop_info,iso,period_data =isolate(pop_info,1,40)
    if parameters == 'Omicron':
        np.save('region_cluster/sample_result_old_omicron1.npy', sample_dict)
        np.save('region_cluster/per_data_old_omicron1.npy', period_data)
    else:
        np.save('region_cluster/sample_result_old1.npy',sample_dict)
        np.save('region_cluster/per_data_old1.npy',period_data)

if __name__ == "__main__":
    main()
