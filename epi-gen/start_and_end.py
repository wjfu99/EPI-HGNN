import os
import numpy as np
files = os.listdir('./result')
first_time=[]
last_time=[]
from tqdm import tqdm
for p in tqdm(files):
    a1 = np.load('./result/'+p,allow_pickle=True).item()
    for i in a1:
        first_time.append(list(a1[i].keys())[0])
        last_time.append(list(a1[i].keys())[-1])
np.save('./first.npy',first_time)
np.save('./last.npy',last_time)