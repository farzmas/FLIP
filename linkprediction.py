import pickle as pk
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

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
# def get_modularity(G,y_pred,communities , grt):
#     modularity_old = nx.community.modularity(G,communities)
#     G_new = G.copy()
#     median = 0.5
#     for i in range(grt.shape[0]):
#         if y_pred[i] > median:
#             G_new.add_edge(grt[i][0],grt[i][1])
#     modularity_new = nx.community.modularity(G_new,communities)  
#     return modularity_new, modularity_old

def get_edge_embeddings(grt, node_emb):
    embs = []
    y = list()
    for i in range(grt.shape[0]):
        row = grt[i]
        emb1 = node_emb[row[0]]
        emb2 =  node_emb[row[1]]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
        if row[2]==-1:
            y.append(0)
        else:
            y.append(row[2])
        
    embs = np.array(embs)
    y = np.array(y)
    return embs, y


def link_predict(node_embd, train , test, return_train_preds=False):      
    train_edge_embs, y_train  =  get_edge_embeddings(train, node_embd)
    test_edge_embs, _  =  get_edge_embeddings(test, node_embd)
    edge_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.03, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)


    #edge_classifier = LogisticRegression(solver='lbfgs',random_state=0)
    edge_classifier.fit(train_edge_embs,  y_train)
    y_test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    y_train_preds = edge_classifier.predict_proba(train_edge_embs)[:, 1]
    return y_train_preds, y_test_preds
 