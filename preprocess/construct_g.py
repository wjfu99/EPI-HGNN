import numpy as np
import pickle as pkl
import scipy.sparse as ss


def generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = ss.csr_matrix(H)
    # n_edge = H.shape[1]
    # the weight of the hyperedge
    # the degree of the node
    DV = np.sum(H, axis=1).getA()
    # the degree of the hyperedge
    DE = np.sum(H, axis=0).getA()

    invDE = ss.diags(np.power(DE, -1).flatten())
    DV2 = np.power(DV, -0.5).flatten()
    # 去除掉DV2中的nan、inf值
    DV2[np.isinf(DV2)] = 0
    DV2[np.isnan(DV2)] = 0
    DV2 = ss.diags(DV2)
    HT = H.T

    return DV2 * H * invDE, HT * DV2


un = 10
rz = True

H = np.load("../privacy/noposterior/H_un={}_rm01={}.npy".format(un, rz))
G1, G2 = generate_G_from_H(H)
G = G1 * G2
# G = ss.coo_matrix(G)
# np.save("../privacy/noposterior/G_un={}_rz={}".format(un, rz), G.todense)
np.save("../privacy/noposterior/G1_un={}_rz={}".format(un, rz), G1.todense())
np.save("../privacy/noposterior/G2_un={}_rz={}".format(un, rz), G2.todense())
