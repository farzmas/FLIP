from Baselines.parameter_parser import parameter_parser
from utils import tab_printer, read_data
from Baselines.proximity_base import jaccard, adamic_adar, preferential_attachment
from Baselines.walk_base import node2vec_alg
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from Baselines.utils import link_predict, get_accuracy_scores, get_modularity, Logger, eq_odds_postproc_baseline, get_second_measure, eqoddpost, greedy_demod, inds_to_ar, good_greedy, add_inds, to_pred
import networkx as nx

def main():
	args = parameter_parser()
	tab_printer(args)
	logger = Logger(args)
	data = read_data(args)
	print(len(data['examples']))
	old_mod = list()
	for i in tqdm(range(len(data['examples'])),'Graph ID:'):
		G = data['examples'][i][0]
		grt = data['examples'][i][1]
		args.single_id = i
		if 'split' in args.file_name:
			train, test = grt['train'], grt['test']
			train_inds = grt['train_idx']
			test_inds = grt['test_idx']
		else:
			indices = np.arange(grt.shape[0])
			train , test= train_test_split(grt, test_size=args.test_size)
		if args.algorithm == "n2v":
			valid = grt['valid']
			y_pred_train, y_pred_valid ,y_pred = node2vec_alg(G, train,valid , test, args)
		elif args.algorithm == "dw":
			if 'split' in args.file_name:
				#valid = grt['valid']
				args.p = 1.0
				args.q = 1.0
				y_pred_train,y_pred_valid, y_pred = node2vec_alg(G, train,None , test, args)
			else:
				args.p = 1.0
				args.q = 1.0
				y_pred_train,_, y_pred = node2vec_alg(G, train,None , test, args)
		elif args.algorithm == "jac":
			print('jaccard')
			y_pred = jaccard(G, test)
			print(y_pred)
			y_pred_train = jaccard(G, train)
		elif args.algorithm == "adar":
			y_pred = adamic_adar(G, test)
			y_pred_train = adamic_adar(G, train)
		elif args.algorithm == "prf":
			y_pred = preferential_attachment(G, test)
			y_pred_train = preferential_attachment(G, train)
		else:
			raise ValueError('The algorithm does not belong to the set of supported algs.')
		old_y_pred = y_pred
		old_y_pred_train = y_pred_train
		if args.postproc == 'eq_odds':
			v_is_prot = np.ones(len(data['G'])) * -1
			for i, community in enumerate(data['communities']):
				inds = list(data['communities'][i])
				v_is_prot[inds] = i
			y_pred_train, y_pred = eqoddpost(y_pred_train, y_pred, train, test, v_is_prot)
		elif args.postproc == 'greedy':
			percent = args.change_percent
			adj = nx.convert_matrix.to_numpy_array(G)
			v_is_prot = np.ones((len(data['G']), 1)) * -1
			for i, community in enumerate(data['communities']):
				inds = list(data['communities'][i])
				v_is_prot[inds, 0] = i

			pred_g = G.copy()
			train_w_pred = train.copy()
			train_w_pred[:, 2] = y_pred_train > np.median(y_pred_train)
			adj_train = nx.convert_matrix.to_numpy_array(add_inds(pred_g, train_w_pred))
			train_pred_inds = inds_to_ar(train, np.zeros_like(adj)).astype(int)
			y_pred_train = good_greedy(adj.astype(int), train_pred_inds, v_is_prot.astype(int), percent)
			y_pred_train = to_pred(y_pred_train, train)

			addG = G.copy()
			addG = add_inds(addG, train)
			if not 'caseStudy' in args.file_name:
				validation = grt['valid']

				pred_g = addG.copy()
				validation_w_pred = validation.copy()
				validation_w_pred[:, 2] = y_pred_valid > np.median(y_pred_train)
				adj_val = nx.convert_matrix.to_numpy_array(add_inds(pred_g, validation_w_pred))
				val_pred_inds = inds_to_ar(validation, np.zeros_like(adj)).astype(int)

				val_pred_inds = inds_to_ar(validation, np.zeros_like(adj)).astype(int)
				y_pred_val = good_greedy(nx.convert_matrix.to_numpy_array(addG).astype(int), val_pred_inds, v_is_prot.astype(int), percent)
				y_pred_val = to_pred(y_pred_val, validation)
				addG = add_inds(addG, validation)

			pred_g = addG.copy()
			test_w_pred = test.copy()
			test_w_pred[:, 2] = y_pred > np.median(y_pred_train)
			adj_test = nx.convert_matrix.to_numpy_array(add_inds(pred_g, test_w_pred))
			test_pred_inds = inds_to_ar(test, np.zeros_like(adj)).astype(int)
			y_pred = good_greedy(nx.convert_matrix.to_numpy_array(pred_g).astype(int), test_pred_inds, v_is_prot.astype(int), percent)
			y_pred = to_pred(y_pred, test.copy())

		if args.algorithm == 'n2v':
			acc_v , roc_auc_v, _ = get_accuracy_scores(valid, y_pred_valid, y_pred_train,args)
			logger.accs_v.append(acc_v)
			logger.aucs_v.append(roc_auc_v,)
		acc , roc_auc, pa_ap = get_accuracy_scores(test, y_pred, y_pred_train,args)
		logger.accs.append(acc)
		logger.aucs.append(roc_auc)
		modularity_new, modularity_ground = get_modularity(G,y_pred,y_pred_train,data['communities'], test, args)
		old_modularity_new, old_modularity_ground = get_modularity(G,old_y_pred,old_y_pred_train,data['communities'], test, args)
		old_mod.append((modularity_ground- old_modularity_new)/np.abs(modularity_ground))
		#m_before, m_after = get_second_measure(G,y_pred,y_pred_train,data['communities'], test, args)
		print('modulairty ground:',modularity_ground)
		print('modulairty after:',modularity_new)
		print('old modulairty after:',old_modularity_new)
		#print('second measure before:', m_before )
		#print('second measure before:', m_after )
		logger.y_hats.append((y_pred_train,y_pred))
		logger.modularities.append((modularity_ground- modularity_new)/np.abs(modularity_ground))
		#logger.modularities.append((modularity_old- modularity_new)/modularity_old)
	logger.averages()
	logger.log_results()
	logger.print_results()
	logger.pickle_results()
	print(np.mean(old_mod))


if __name__ =="__main__":
	main()