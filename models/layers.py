import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(G, x)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class HGNN_t_conv(nn.Module):
    def __init__(self, in_ft, out_ft, embedding_dim=4, time_slot=7 * 24, bias=True):
        super(HGNN_t_conv, self).__init__()
        self.embedding_dim = embedding_dim
        # self.weight_parameter = Parameter(torch.Tensor(in_ft/embedding_dim, in_ft/embedding_dim))
        self.weight_parameter = Parameter(torch.Tensor(time_slot, time_slot))
        # self.unname = torch.block_diag([1, 1, 1, 1] for i in range(7*24))
        self.u = np.zeros((time_slot, time_slot * 4))
        self.v = np.zeros((4 * time_slot, 4 * time_slot))
        for i in range(time_slot):
            self.u[i, 4 * i:4 * (i + 1)] = 1
            self.v += np.eye(4 * time_slot, k=4 * i)
        self.u = torch.Tensor(self.u).to(torch.device('cuda:3'))
        self.v = torch.Tensor(self.v).to(torch.device('cuda:3'))

        # self.weight = torch.Tensor(in_ft, in_ft).to(torch.device('cuda:3'))

        if bias:
            self.bias = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_parameter.size(1))
        self.weight_parameter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # for i in range(self.weight_parameter.size(0)):
        #     for j in range(0, i+1):
        #         self.weight[i:i+self.embedding_dim, j:j+self.embedding_dim] = self.weight_parameter[i, j]
        # print('erqwrqwrqwr', x.size(1), self.u.t().size(0))
        w = torch.matmul(self.u.t(), self.weight_parameter)
        w = torch.matmul(w, self.u)
        w = torch.mul(w, self.v)
        x = torch.matmul(x, w)

        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(G, x)
        return x



class HGNN_t_conv_v2(nn.Module):
    def __init__(self, in_ft, out_ft, embedding_dim=4, time_slot=7 * 24, bias=True):
        super(HGNN_t_conv_v2, self).__init__()
        self.embedding_dim = embedding_dim
        # self.weight_parameter = Parameter(torch.Tensor(in_ft/embedding_dim, in_ft/embedding_dim))
        self.weight_parameter = Parameter(torch.Tensor(time_slot, time_slot))
        self.weight_t_parameter = Parameter(torch.Tensor(embedding_dim, embedding_dim))

        # self.unname = torch.block_diag([1, 1, 1, 1] for i in range(7*24))
        self.u = np.zeros((time_slot, time_slot * embedding_dim))
        self.v = np.zeros((embedding_dim * time_slot, embedding_dim * time_slot))
        self.u1 = np.zeros((embedding_dim * time_slot, embedding_dim))
        self.v1 = np.zeros((embedding_dim * time_slot, embedding_dim * time_slot))
        dw = np.array(range(1, time_slot+1))
        dw = dw.repeat(embedding_dim)
        # dw = np.count_nonzero(w2, axis=0)
        dw = dw.astype(np.float)
        dw = np.power(dw, -1)
        dw = np.diag(dw)
        self.dw = torch.Tensor(dw).to(torch.device('cuda:3'))
        for i in range(time_slot):
            self.u[i, embedding_dim * i:embedding_dim * (i + 1)] = 1
            self.v += np.eye(embedding_dim * time_slot, k=embedding_dim * i)
            self.u1[i * embedding_dim:(i + 1) * embedding_dim, :] += np.identity(embedding_dim)
            self.v1[i * embedding_dim:(i + 1) * embedding_dim, i * embedding_dim:(i + 1) * embedding_dim] = 1
        self.u = torch.Tensor(self.u).to(torch.device('cuda:3'))
        self.v = torch.Tensor(self.v).to(torch.device('cuda:3'))
        self.u1 = torch.Tensor(self.u1).to(torch.device('cuda:3'))
        self.v1 = torch.Tensor(self.v1).to(torch.device('cuda:3'))

        # self.weight = torch.Tensor(in_ft, in_ft).to(torch.device('cuda:3'))

        if bias:
            self.bias = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_parameter.size(1))
        self.weight_parameter.data.uniform_(-stdv, stdv)
        self.weight_t_parameter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # for i in range(self.weight_parameter.size(0)):
        #     for j in range(0, i+1):
        #         self.weight[i:i+self.embedding_dim, j:j+self.embedding_dim] = self.weight_parameter[i, j]
        # print('erqwrqwrqwr', x.size(1), self.u.t().size(0))
        w = torch.matmul(self.u.t(), self.weight_parameter)
        w = torch.matmul(w, self.u)
        w = torch.mul(w, self.v)
        w_t = torch.matmul(self.u1, self.weight_t_parameter)
        w_t = torch.matmul(w_t, self.u1.t())
        w_t = torch.mul(w_t, self.v1)

        x = torch.matmul(x, w)
        x = torch.matmul(x, self.dw)
        x = torch.matmul(x, w_t)

        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(G, x)
        return x


class HGNN_t_conv_v3(nn.Module):
    def __init__(self, in_ft, out_ft, embedding_dim=4, time_slot=40*48, bias=True):
        super(HGNN_t_conv_v3, self).__init__()
        self.embedding_dim = embedding_dim
        self.weight_parameter = Parameter(torch.Tensor(time_slot*embedding_dim, time_slot*embedding_dim))
        self.u = np.kron(np.triu(np.ones((time_slot, time_slot))), np.ones((embedding_dim, embedding_dim)))
        self.u = torch.Tensor(self.u)  # .to(torch.device('cuda:3'))

        if bias:
            self.bias = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_parameter.size(1))
        self.weight_parameter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):

        w = torch.mul(self.weight_parameter, self.u)
        x = torch.matmul(x, w)
        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(G, x)
        return x

# 基于折中算法的对element wise进行改进
class HGNN_t_conv_v4(nn.Module):
    def __init__(self, in_ft, out_ft, embedding_dim=10, time_slot=40*48, bias=True):
        super(HGNN_t_conv_v4, self).__init__()
        self.embedding_dim = embedding_dim
        # self.weight_parameter = Parameter(torch.Tensor(time_slot*embedding_dim, time_slot*embedding_dim))
        self.weight_parameter = Parameter(torch.Tensor(in_ft, in_ft))
        # self.u = np.kron(np.triu(np.ones((time_slot, time_slot))), np.ones((embedding_dim, embedding_dim)))
        # self.u = torch.Tensor(self.u).to(torch.device('cuda:4'))
        if bias:
            self.bias = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_parameter.size(1))
        self.weight_parameter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, ht_m: torch.Tensor, u: torch.Tensor):

        w = torch.mul(self.weight_parameter, u)
        # w = self.weight_parameter
        x = torch.matmul(x, w)
        x = torch.mm(g2, x)
        x = torch.mul(ht_m, x)
        x = torch.mm(g1, x)
        if self.bias is not None:
            x = x + self.bias
        return x


class HGNN_t_conv_v5(nn.Module):
    def __init__(self, t, unit_size_x, unit_size_y, unit_num, bias=True):
        super(HGNN_t_conv_v5, self).__init__()
        # self.weight_parameter = Parameter(torch.Tensor(time_slot*embedding_dim, time_slot*embedding_dim))
        self.weight_parameter = Parameter(torch.Tensor(unit_num*unit_size_y, unit_num*unit_size_x))
        self.u = np.kron(np.triu(np.ones((unit_num, unit_num))), np.ones((unit_size_y, unit_size_x)))
        self.u = torch.Tensor(self.u).cuda()
        # make mask of HT
        ht_m = np.zeros((t.shape[0], unit_num * unit_size_x), dtype=np.int32)
        for i, ti in enumerate(t):
            ht_m[i, :int((ti[1] + 1) / (4 * 48) * unit_size_x)] = 1
        self.ht_m = ht_m
        self.ht_m = torch.Tensor(self.ht_m).cuda()
        # self.u = np.kron(np.triu(np.ones((time_slot, time_slot))), np.ones((embedding_dim, embedding_dim)))
        # self.u = torch.Tensor(self.u).to(torch.device('cuda:4'))
        if bias:
            self.bias = Parameter(torch.Tensor(unit_num*unit_size_x))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_parameter.size(1))
        self.weight_parameter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor):

        w = torch.mul(self.weight_parameter, self.u)
        # w = self.weight_parameter
        x = torch.matmul(x, w)
        x = torch.mm(g2, x)
        x = torch.mul(self.ht_m, x)
        x = torch.mm(g1, x)
        if self.bias is not None:
            x = x + self.bias
        return x

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x
