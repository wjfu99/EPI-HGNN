

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
import pickle as pkl

embedding_dim = 4
time_interval = 4
trace_array = np.load('../privacy/noposterior/trace_array_{}h.npy'.format(time_interval))
user_num = trace_array.shape[0]
region_num = trace_array.max() + 1
time_slot = trace_array.shape[1]

trace_array += 1


fts = torch.Tensor(trace_array).long()
embedding = torch.nn.Embedding(region_num+1, embedding_dim, padding_idx=None)
fts = embedding(fts)
fts1 = torch.reshape(fts, (user_num, time_slot*embedding_dim))
fts = fts.detach().numpy()
fts1 = fts1.detach().numpy()
w = embedding.weight

# with open('cluster/data_afcluster/fts', 'wb') as f:
#     pkl.dump(fts1, f)
np.save('../privacy/noposterior/fts_{}h_emb={}.npy'.format(time_interval, embedding_dim), fts1)