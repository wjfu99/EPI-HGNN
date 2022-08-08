import numpy as np
import torch
import pickle as pkl
from scipy.linalg import block_diag

# #创建一个二维张量
# x =torch.randn((2,3)) #torch.Size([2, 3])
# #在x的两个维度上分别重复4次，2次，即得目标维度：4x2=8, 2x3=6，即（8，6）
# y = torch.Tensor(2, 2)
# y[:, :] = x[0, 0]
# x[0, 0] = 1
# a = 1
# for i in range(0, 10):
#     print(i)
#
# def abd():
#     global a
#     a = 2
# b = [1, 2, 3]
# a = [b for i in range(10)]
# c = torch.block_diag(torch.tensor([[1, 1, 1, 1]]))
with open("../datasets/fts", 'rb') as f:
    fts = pkl.load(f)

time_slot = 7*24
u = np.zeros((time_slot, time_slot*4))
v = np.zeros((4*time_slot, 4*time_slot))
u1 = np.zeros((4*time_slot, 4))
v1 = np.zeros((4*time_slot, 4*time_slot))
w_t = np.random.rand(4, 4)
# w_new = block_diag(w_t_p, w_t_p, w_t_p, w_t_p, w_t_p)
for i in range(time_slot):
    u[i, 4*i:4*(i+1)] = 1
    u1[i*4:(i+1)*4, :] += np.identity(4)
    v += np.eye(4 * time_slot, k=4 * i)
    v1[i*4:(i+1)*4, i*4:(i+1)*4] = 1
w = np.random.rand(time_slot, time_slot)
w1 = u.T.dot(w).dot(u)
w2 = w1 * v
dw = np.array(range(1, 169))
dw = dw.repeat(4)
# dw = np.count_nonzero(w2, axis=0)
dw = dw.astype(np.float)
dw = np.power(dw, -1)
dw = np.diag(dw)
w3 = u1.dot(w_t).dot(u1.T)
w3 = w3 * v1

u2 = np.ones((4, 4))
v2 = np.triu(np.ones((time_slot, time_slot)))
w4 = np.kron(v2, u2)

