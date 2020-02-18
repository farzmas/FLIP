import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from Baselines.utils import get_edge_embeddings, walk_exist, save_walks, load_walks
from node2vec import Node2Vec
from gensim.models import Word2Vec


def node2vec_alg(G, train, valid , test, args):  
	if walk_exist(args):
		print('the walk exist')
		walks = load_walks(args)
		model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=1, workers= args.workers, iter=args.epochs)
	else:
		node2vec= Node2Vec(G, dimensions=args.dimensions, walk_length=args.len_of_walks, 
					   num_walks=args.number_of_walks, p= args.p, q= args.q,workers= args.workers)
		save_walks(node2vec.walks, args)
		model = node2vec.fit( size=args.dimensions, window=args.window_size, min_count=1, workers=args.workers, iter=args.epochs)
	train_edge_embs, y_train  =  get_edge_embeddings(train, model.wv)
	test_edge_embs, _  =  get_edge_embeddings(test, model.wv)  
	if 'split' in args.file_name and valid is not None:
		valid_edge_embs, _  =  get_edge_embeddings(valid, model.wv)
	edge_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.03, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
	edge_classifier.fit(train_edge_embs,  y_train) 
	y_train_preds =  edge_classifier.predict_proba(train_edge_embs)[:, 1]
	if 'split' in args.file_name and valid is not None:
		y_valid_preds =  edge_classifier.predict_proba(valid_edge_embs)[:, 1]
	else:
		y_valid_preds = None
	y_test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
	return y_train_preds,y_valid_preds ,y_test_preds 


