import pickle as pk
import numpy as np
import networkx as nx
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import pandas as pd
from itertools import product

from equalized_odds_and_calibration.eq_odds import Model
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset

def second_measure(G, communities):
    coms = dict()
    for ct,com in enumerate( communities):
        for node in com:
            coms[node]= ct
    n = len(G.nodes())
    m = len(G.edges())
    p_link = m/(n*(n-1)/2)
    p_link_label_eq = 0
    for e in G.edges():
        if coms[e[0]]==coms[e[1]]:
            p_link_label_eq+=1
    p_link_label_eq = p_link_label_eq/(n*(n-1)/2)
    p_label_eq = 0
    for ct,com in enumerate(communities):
        p_label_eq += len(com)*(len(com)-1)/2
    p_label_eq = p_label_eq/ (n*(n-1)/2)  
    return round(p_link,4), round( p_link_label_eq/p_label_eq,4)
def get_second_measure(G,y_pred,y_pred_train,communities , grt,args):
    pr_link_before, pr_link_same_before = second_measure(G, communities)
    G_new = G.copy()
    median = np.median(y_pred_train)
    if args.algorithm == "n2v" or args.algorithm == "dw":
        median = 0.5
    for i in range(grt.shape[0]):
        if y_pred[i] > median:
            G_new.add_edge(grt[i][0],grt[i][1])
    pr_link_after, pr_link_same_after = second_measure(G_new, communities)
    return (pr_link_before, pr_link_same_before), (pr_link_after, pr_link_same_after)
    
def get_accuracy_scores(grt, y_pred, y_pred_train,args, is_bin=False):
    # check if -1 is used for  negatives label instead of 0.
    idx = grt[:,2]==-1
    grt[idx,2] = 0
    # create the acctual labels vector
    y_true = grt[:,2]
    # calcualte the accuracy 
    roc_score = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    median = np.median(y_pred_train)
    if args.algorithm == "n2v" or args.algorithm == "dw" or is_bin:
        median = 0.5
    acc = accuracy_score(y_true, y_pred > median)
    return acc, roc_score,  ap_score

def get_modularity(G,y_pred,y_pred_train,communities , grt,args, is_bin=False):
    G_ground = G.copy()
    G_new = G.copy()
    
    thresh = np.median(y_pred_train)
    if args.algorithm == "n2v" or args.algorithm == "dw":
        thresh = 0.5
    if is_bin:
        thresh = 0.01
    for i in range(grt.shape[0]):
        if y_pred[i] > thresh:
            G_new.add_edge(grt[i][0],grt[i][1])
        if grt[i][2]==1:
            G_ground.add_edge(grt[i][0],grt[i][1])
    modularity_new = nx.community.modularity(G_new,communities) 
    modularity_ground = nx.community.modularity(G_ground,communities)  
    return modularity_new, modularity_ground


def get_edge_embeddings(grt, node_emb):
    embs = []
    y = list()
    for i in range(grt.shape[0]):
        row = grt[i]
        emb1 = node_emb[str(row[0])]
        emb2 =  node_emb[str(row[1])]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
        if row[2]==-1:
            y.append(0)
        else:
            y.append(row[2])
        
    embs = np.array(embs)
    y = np.array(y)
    return embs, y

def walk_exist(args):
	path = args.walk_path+str(args.single_id) +'_n2v_'+args.file_name.split('.')[0]+'p='+str(args.p) +'q='+str(args.q)+'.txt'
	#print(path)
	return os.path.isfile(path)

def load_walks(args):
	path = args.walk_path+str(args.single_id) +'_n2v_'+args.file_name.split('.')[0]+'p='+str(args.p) +'q='+str(args.q)+'.txt'
	file_ = open(path,'r')
	walks = list()
	for line in file_:
		if len(line)>2:
			walk = line.strip().split()
			walks.append(walk)
	return walks
def save_walks(walks_str,args):
	path = args.walk_path +str(args.single_id)+'_n2v_'+args.file_name.split('.')[0]+'p='+str(args.p) +'q='+str(args.q)+'.txt'
	file_ = open(path,'a')
	for walk in walks_str:
		line = str()
		for node in walk:
			line += node+ ' '
		line += '\n'
		file_.write(line)
	file_.close()

def link_predict(node_embd, train , test):      
    train_edge_embs, y_train  =  get_edge_embeddings(train, node_embd)
    test_edge_embs, _  =  get_edge_embeddings(test, node_embd)    
    edge_classifier = LogisticRegression(solver='lbfgs',random_state=0)
    edge_classifier.fit(train_edge_embs,  y_train)    
    y_test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    return y_test_preds 

class Logger():
	def __init__(self,args):
		self.args = args
		self.modularities = list()
		self.accs = list()
		self.aucs = list()
		if self.args.algorithm =='n2v':
			self.accs_v = list()
			self.aucs_v = list()
		self.y_hats = list()
		self.second_measure = list() 
	def averages(self):
		self.avg_modularity =(np.round(np.mean(self.modularities),decimals=4),
							  np.round(np.var(self.modularities),decimals=4))
		self.avg_acc = (np.round(np.mean(self.accs),decimals=4), np.round(np.var(self.accs),decimals=4))
		self.avg_auc = (np.round(np.mean(self.aucs),decimals=4), np.round(np.var(self.aucs),decimals=4))
		if self.args.algorithm =='n2v':
			self.avg_acc_v = (np.round(np.mean(self.accs_v),decimals=4), np.round(np.var(self.accs_v),decimals=4))
			self.avg_auc_v = (np.round(np.mean(self.aucs_v),decimals=4), np.round(np.var(self.aucs_v),decimals=4))
	def log_results(self,single = False):
		if single:
			file = open(self.args.log_path + str(self.args.single_id) +self.args.log_file,'w')
		else:
			file = open(self.args.log_path  +self.args.log_file,'w')
		file.write('modularity_reduction '+str(self.avg_modularity[0]) + ' +/- '+ str(self.avg_modularity[1])+'\n')
		file.write('acc '+str(self.avg_acc[0]) + ' +/- '+ str(self.avg_acc[1])+'\n')
		file.write('auc '+str(self.avg_auc[0])+ ' +/- '+ str(self.avg_auc[1])+'\n')
		if self.args.algorithm == 'n2v':
			file.write('acc_v '+str(self.avg_acc_v[0]) + ' +/- '+ str(self.avg_acc_v[1])+'\n')
			file.write('auc_v '+str(self.avg_auc_v[0])+ ' +/- '+ str(self.avg_auc_v[1])+'\n')
		file.close()
	def print_results(self):
		print('average modularity reduction = ',self.avg_modularity[0] , '+/-', self.avg_modularity[1])
		print('average acc = ', self.avg_acc[0], '+/-', self.avg_acc[1])
		print('average auc = ', self.avg_auc[0], '+/-', self.avg_auc[1])
		if self.args.algorithm == 'n2v':
			print('average acc_v = ', self.avg_acc_v[0], '+/-', self.avg_acc_v[1])
			print('average auc_v = ', self.avg_auc_v[0], '+/-', self.avg_auc_v[1])
	def pickle_results(self, single = False):
		pickle_file = self.args.log_file.split('.')[0]+'.pk'
		if single:
			file = open(self.args.log_path + str(self.args.single_id) +pickle_file,'wb')
		else: 
			file = open(self.args.log_path  +pickle_file,'wb')
		pk.dump(self, file )
		file.close()

def eq_odds_postproc(y_pred_train, y_pred_test, data, exp, train_inds, test_inds):
	prot = np.ones(len(data['G'].node)) * -1
	for i, community in enumerate(data['communities']):
		inds = list(data['communities'][i])
		prot[inds] = i

	v1 = data['examples'][exp][1][:, 0]
	v2 = data['examples'][exp][1][:, 1]
	prot_edge = (prot[v1] == prot[v2]) * 1.
	targs = data['examples'][0][1][:, 2] * 0.5 + 0.5

	df_true_train = pd.DataFrame(np.zeros_like(train_inds))
	df_true_train['protected'] = prot_edge[train_inds]
	df_true_train['targs'] = targs[train_inds]

	df_pred_train = pd.DataFrame(np.zeros_like(train_inds))
	df_pred_train['protected'] = prot_edge[train_inds]
	df_pred_train['targs'] = (y_pred_train > 0.5) * 1.

	df_pred_test = pd.DataFrame(np.zeros_like(test_inds))
	df_pred_test['protected'] = prot_edge[test_inds]
	df_pred_test['targs'] = (y_pred_test > 0.5) * 1.

	dataset_true_train = BinaryLabelDataset(df=df_true_train, label_names=['targs'], protected_attribute_names=['protected'])
	dataset_pred_train = BinaryLabelDataset(df=df_pred_train, label_names=['targs'], protected_attribute_names=['protected'])
	dataset_pred_test = BinaryLabelDataset(df=df_pred_test, label_names=['targs'], protected_attribute_names=['protected'])

	privileged_groups = [{'protected': 1}]
	unprivileged_groups = [{'protected': 0}]

	postproc = EqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups, seed=0)
	postproc = postproc.fit(dataset_true_train, dataset_pred_train)

	out = postproc.predict(dataset_pred_test)
	return np.squeeze(out.labels)

def edge_prot(v_is_prot, v1, v2):
	return (v_is_prot[v1] == v_is_prot[v2]) * 1.

def eq_odds_postproc_baseline(y_pred_train, y_pred_test, data, exp):
	train_inds = data['examples'][exp][1]['train_idx']
	test_inds = data['examples'][exp][1]['test_idx']
	v_is_prot = np.ones(len(data['G'])) * -1
	for i, community in enumerate(data['communities']):
		inds = list(data['communities'][i])
		v_is_prot[inds] = i

	exp_data = data['examples'][exp][1]
	df_true_train = pd.DataFrame(np.zeros_like(train_inds))
	df_true_train['protected'] = edge_prot(v_is_prot, exp_data['train'][:, 0], exp_data['train'][:, 1])
	df_true_train['targs'] = exp_data['train'][:, 2]

	df_pred_train = pd.DataFrame(np.zeros_like(train_inds))
	df_pred_train['protected'] = edge_prot(v_is_prot, exp_data['train'][:, 0], exp_data['train'][:, 1])
	df_pred_train['targs'] = (y_pred_train > np.median(y_pred_train)) * 1.

	df_pred_test = pd.DataFrame(np.zeros_like(test_inds))
	df_pred_test['protected'] = edge_prot(v_is_prot, exp_data['test'][:, 0], exp_data['test'][:, 1])
	df_pred_test['targs'] = (y_pred_test > np.median(y_pred_train)) * 1.

	dataset_true_train = BinaryLabelDataset(df=df_true_train, label_names=['targs'], protected_attribute_names=['protected'])
	dataset_pred_train = BinaryLabelDataset(df=df_pred_train, label_names=['targs'], protected_attribute_names=['protected'])
	dataset_pred_test = BinaryLabelDataset(df=df_pred_test, label_names=['targs'], protected_attribute_names=['protected'])

	privileged_groups = [{'protected': 1}]
	unprivileged_groups = [{'protected': 0}]

	postproc = EqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups, seed=0)
	postproc = postproc.fit(dataset_true_train, dataset_pred_train)

	out = postproc.predict(dataset_pred_test)
	return np.squeeze(out.labels)

def eqoddpost(y_pred_train, y_pred, train, test, v_is_prot):
    ybin_train = train[:, 2]
    ybin_test = test[:, 2]
    
    clf = LogisticRegression()
    clf = clf.fit(y_pred_train.reshape(-1, 1), ybin_train)
    preds_train = clf.predict_proba(y_pred_train.reshape(-1, 1))[:, 1]
    preds_test = clf.predict_proba(y_pred.reshape(-1, 1))[:, 1]

    a = edge_prot(v_is_prot, train[:, 0], train[:, 1])
    a_test = edge_prot(v_is_prot, test[:, 0], test[:, 1])
    
    df_train = pd.DataFrame()
    df_train['group'] = a
    df_train['label'] = ybin_train
    df_train['prediction'] = preds_train

    df_test = pd.DataFrame()
    df_test['group'] = a_test
    df_test['label'] = ybin_test
    df_test['prediction'] = preds_test
    
    train0_inds = df_train['group'] == 0
    train1_inds = df_train['group'] == 1
    test0_inds = df_test['group'] == 0
    test1_inds = df_test['group'] == 1

    group_0_val_data = df_train[train0_inds]
    group_1_val_data = df_train[train1_inds]
    group_0_test_data = df_test[test0_inds]
    group_1_test_data = df_test[test1_inds]

    group_0_val_model = Model(group_0_val_data['prediction'].as_matrix(), group_0_val_data['label'].as_matrix())
    group_1_val_model = Model(group_1_val_data['prediction'].as_matrix(), group_1_val_data['label'].as_matrix())
    group_0_test_model = Model(group_0_test_data['prediction'].as_matrix(), group_0_test_data['label'].as_matrix())
    group_1_test_model = Model(group_1_test_data['prediction'].as_matrix(), group_1_test_data['label'].as_matrix())

    # Find mixing rates for equalized odds models
    eq_odds_group_0_train_model, eq_odds_group_1_train_model, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

    # Apply the mixing rates to the test models
    eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                           group_1_test_model,
                                                                           mix_rates)

    new_train_preds = np.zeros_like(ybin_train) * 1.
    new_test_preds = np.zeros_like(ybin_test) * 1.

    new_train_preds[train0_inds.values] += eq_odds_group_0_train_model.pred
    new_train_preds[train1_inds.values] += eq_odds_group_1_train_model.pred
    new_test_preds[test0_inds] += eq_odds_group_0_test_model.pred
    new_test_preds[test1_inds] += eq_odds_group_1_test_model.pred
    return new_train_preds, new_test_preds

def greedy_demod(data, exp, train, y_pred_train, exponent, xtraedges=None):
    base = data['examples'][exp][0].copy()
    if not xtraedges is None:
        for inds in xtraedges:
            base = add_inds(base, inds)
    baset = add_inds(base.copy(), train)
    base = nx.convert_matrix.to_numpy_array(base)
    baset = nx.convert_matrix.to_numpy_array(baset)

    row_sum = np.sum(baset, axis=1, keepdims=True)
    col_sum = np.sum(baset, axis=0, keepdims=True)
    d = np.dot(row_sum, col_sum)

    v_is_prot = np.ones(len(data['G'])) * -1
    for i, community in enumerate(data['communities']):
        inds = list(data['communities'][i])
        v_is_prot[inds] = i

    is_prot = np.zeros_like(d)
    for ind in product(list(range(v_is_prot.shape[0])), list(range(v_is_prot.shape[0]))):
        is_prot[ind] = (v_is_prot[ind[0]] == v_is_prot[ind[1]])*1.
    remaining_d = np.multiply(np.multiply(d, np.multiply(baset, (1 - base))), (is_prot))

    train_median = np.median(y_pred_train)
    train_pred = train.copy()
    train_pred[:, 2] = (y_pred_train > train_median) * 1.
    pred_adj = nx.convert_matrix.to_numpy_array(add_inds(nx.from_numpy_array(base), train_pred))

    removed_inds = list()

    alpha = 10**-exponent
    beta = 10**exponent

    to_remove = 100
    best_loss = 9.9999 * 10**5
    best_d = remaining_d.copy()
    cur_adj = pred_adj.copy()
    best_adj = cur_adj.copy()
    while True:
        remove_ind = np.unravel_index(np.argmax(remaining_d), remaining_d.shape)
        remaining_d[remove_ind] = 0
        cur_adj[remove_ind] = 0

        cur_loss = loss(pred_adj, cur_adj, data['communities'], beta, alpha)
        if cur_loss <= best_loss:
            removed_inds.append(remove_ind)
            best_loss = cur_loss
            best_d = remaining_d.copy()
            best_adj = cur_adj.copy()
        else:
            break
    return best_adj[train[:, 0], train[:, 1]]

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
    '''
    score_other = np.zeros_like(d)
    d_sum1 = np.sum(d[a > 0])  attribute = 1
    d_sum2 = np.sum(d[(1 - a) > 0]) #compute degree sum for all vertices with protected attribute = 0
    score_other[a > 0] = d_sum1
    score_other[(1 - a) > 0] = d_sum2
    score_other -= d # Make sure not to count each vertex's self in its class degree sum
    '''

    #compute class degree sum for each edge (sum of that edge's vertices)
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

def loss(start, end, communities, beta, alpha=1.):
    return alpha * np.mean(np.square(start - end)) + beta * modularity(end, communities)

def modularity(adj, c):
    G = nx.from_numpy_array(adj)
    return nx.community.modularity(G,c)

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