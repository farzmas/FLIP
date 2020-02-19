from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import numpy as np
import networkx as nx

def get_accuracy_scores(grt, y_pred, median = 0.5):
    # check if -1 is used for  negatives label instead of 0.
    idx = grt[:,2]==-1
    grt[idx,2] = 0
    # create the acctual labels vector
    y_true = grt[:,2]
    # calcualte the accuracy 
    roc_score = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred > median)
    return acc, roc_score,  ap_score

def get_modularity(G,y_pred,communities , grt, median =0.5):
    G_ground = G.copy()
    G_new = G.copy()
    for i in range(grt.shape[0]):
        if y_pred[i] > median:
            G_new.add_edge(grt[i][0],grt[i][1])
        if grt[i][2]==1:
            G_ground.add_edge(grt[i][0],grt[i][1])
    modularity_new = nx.community.modularity(G_new,communities) 
    modularity_ground = nx.community.modularity(G_ground,communities)  
    return modularity_new, modularity_ground