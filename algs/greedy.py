import pickle as pk
import numpy as np
import networkx as nx
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


def greedy_wrapper(G, communities, tru_vals, y_pred_train, binarize_thresh, percent, inds_to_add=[]):
    v_is_prot = np.ones((G.number_of_nodes(), 1)) * -1
    for i, community in enumerate(communities):
        inds = list(community)
        v_is_prot[inds, 0] = i

    pred_g = G.copy()

    train_w_pred = tru_vals.copy()
    train_w_pred[:, 2] = y_pred_train > binarize_thresh
    adj_train = nx.convert_matrix.to_numpy_array(add_inds(pred_g, train_w_pred))
    train_pred_inds = inds_to_ar(tru_vals, np.zeros_like(adj_train)).astype(int)

    y_pred_train = good_greedy(adj_train.astype(int).copy(), train_pred_inds.copy(), v_is_prot.astype(int).copy(), percent)
    y_pred_train = to_pred(y_pred_train, tru_vals)
    return y_pred_train


def good_greedy(adj, pred_inds, a, percent):
    # make sure a is a column vector w/ length = number of vertices in graph
    assert len(a.shape) == 2
    assert a.shape[1] == 1
    assert adj.shape[0] == a.shape[0]

    d = np.sum(adj, axis=1, keepdims=True)
    m = np.sum(adj) / 2

    score_pair = np.multiply((d + d.T - 1) / (2 * m) - 1, (a == a.T))
    
    #compute degree sum for all vertices with each protected
    score_other = np.zeros_like(d)
    class_d_sum = {}
    for c in np.unique(a):
        class_d_sum[c] = np.sum(d[a == c])
        score_other[a == c] = class_d_sum[c]
        score_other -= d

    score_other = score_other + score_other.T - np.diag(np.squeeze(score_other))

    score_other = score_other / (2 * m)
    
    score = score_pair + score_other
    
    score[(1 - adj) > 0] *= -1
    score[(1 - pred_inds) > 0] = 9999999
    
    mod_percent = percent * pred_inds.sum() / (adj.size)

    thresh = np.quantile(score, mod_percent)
    flip_inds = score < thresh
    
    adj[flip_inds] = 1 - adj[flip_inds]
    return adj



def add_inds(G, inds):
    n_t = inds.shape[0]
    for j in range(n_t):
        if inds[j][2]==1:
            G.add_edge(inds[j][0], inds[j][1])
    return G

def inds_to_ar(inds, arr):
    n_t = inds.shape[0]
    for j in range(n_t):
        if inds[j][2]==1:
            arr[inds[j][0], inds[j][1]] = 1
            arr[inds[j][1], inds[j][0]] = 1
    return arr

def to_pred(pred_adj, inds):
    pred = np.zeros(inds.shape[0])
    for i in range(pred.shape[0]):
        pred[i] = pred_adj[inds[i, 0], inds[i, 1]]
    return pred