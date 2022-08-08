import pickle as pkl
import numpy as np
from tqdm import tqdm

per_data1 = np.load('../region_cluster/per_data.npy', allow_pickle=True).item()
per_data = np.load('../region_cluster/per_data_old.npy', allow_pickle=True).item()
per_data_o = np.load('../region_cluster/per_data_old_omicron.npy', allow_pickle=True).item()
sample_result = np.load('../region_cluster/sample_result.npy', allow_pickle=True).item()
label = [per_data[0][x]['state'] for x in per_data[0]]
label = np.array(label).reshape(-1, 1)
label1 = [per_data1[0][x]['state'] for x in per_data1[0]]
label1 = np.array(label1).reshape(-1, 1)
label_o = [per_data_o[0][x]['state'] for x in per_data_o[0]]
label_o = np.array(label_o).reshape(-1, 1)
# np.save("../privacy/label_omicron.npy", label)