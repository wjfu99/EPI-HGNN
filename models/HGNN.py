from torch import nn
import torch
from models import HGNN_conv, GraphConvolution, HGNN_t_conv
from models.layers import HGNN_t_conv_v2, HGNN_t_conv_v3, HGNN_t_conv_v4, GraphAttentionLayer, HGNN_t_conv_v5
import torch.nn.functional as F
import math

# class HGNN(nn.Module):
#     def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
#         super(HGNN, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid)
#         self.hgc2 = nn.Linear(n_hid, n_hid)
#         self.hgc3 = nn.Linear(n_hid, n_class)
#
#
#     def forward(self, x, G):
#         x = F.relu(self.hgc1(x, G))
#         x = F.dropout(x, self.dropout)
#         x = F.relu(self.hgc2(x))
#         x = F.dropout(x, self.dropout)
#         x = self.hgc3(x)
#         return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, in_ch)
        self.hgc2 = HGNN_conv(in_ch, in_ch)
        self.hgc3 = nn.Linear(in_ch, n_class)


    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc3(x)
        return x





# 加入embedding层
class HGNN_time_3(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, time_slot=40*48, embedding_dim=10, embedding_num=11459):
        super(HGNN_time_3, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(embedding_num+1, embedding_dim, padding_idx=0)
        self.hgc1 = HGNN_t_conv_v4(in_ch, in_ch, time_slot=time_slot, embedding_dim=embedding_dim)
        self.hgc2 = HGNN_t_conv_v4(in_ch, in_ch, time_slot=time_slot, embedding_dim=embedding_dim)
        self.hgc3 = nn.Linear(in_ch, n_class)


    def forward(self, x, G1, G2, ht_m, u):
        x = F.relu(self.embedding(x))
        # x = torch.reshape(x, (15279, 10, 24, -1))
        # x = torch.mean(x, 2)
        x = torch.reshape(x, (15279, -1))
        x = F.relu(self.hgc1(x, G1, G2, ht_m, u))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G1, G2, ht_m, u))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x)
        return x

# 要做在超边的时间粒度和特征的粒度一样，
class HGNN_time(nn.Module):
    def __init__(self, n_class, t, k, dropout, unit_num, unit_size, embedding_dim, embedding_num=11459):
        super(HGNN_time, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(embedding_num+1, embedding_dim, padding_idx=0)
        self.hgc1 = HGNN_t_conv_v5(t, unit_size_x=k[0], unit_size_y=unit_size*embedding_dim, unit_num=unit_num)
        self.hgc2 = HGNN_t_conv_v5(t, unit_size_x=k[1], unit_size_y=k[0], unit_num=unit_num)
        self.hgc3 = nn.Linear(unit_num*k[1], n_class)


    def forward(self, x, G1, G2):
        x = F.relu(self.embedding(x))
        # x = torch.reshape(x, (15279, 10, 24, -1))
        # x = torch.mean(x, 2)
        x = torch.reshape(x, (15279, -1))
        x = F.relu(self.hgc1(x, G1, G2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G1, G2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x)
        return x

# class GCN(nn.Module):
#     def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(in_ch, n_hid)
#         self.gc2 = GraphConvolution(n_hid, n_class)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout)
#         x = self.gc2(x, adj)
#         return x

class GCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout, embedding_num, embedding_dim):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(embedding_num + 1, embedding_dim, padding_idx=0)
        self.gc1 = GraphConvolution(in_ch, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        # self.gc3 = nn.Linear(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.embedding(x))
        x = torch.reshape(x, (15279, -1))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, embedding_num, embedding_dim):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(embedding_num + 1, embedding_dim, padding_idx=0)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.relu(self.embedding(x))
        x = torch.reshape(x, (15279, -1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, embedding_dim, embedding_num):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.embedding = nn.Embedding(embedding_num + 1, embedding_dim, padding_idx=0)
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = F.relu(self.embedding(X))
        X = X.unsqueeze(0)
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4


