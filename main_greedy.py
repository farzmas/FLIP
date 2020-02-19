from utils.parameter_parser import parameter_parser
from utils.utils import tab_printer, read_data
from utils.eval import get_accuracy_scores, get_modularity
from utils.logger import Logger
from algs.proximity_base import jaccard, adamic_adar, preferential_attachment
from algs.greedy import greedy_wrapper

from tqdm import tqdm
import numpy as np
import networkx as nx




def main():
	args = parameter_parser()
	tab_printer(args)
	data = read_data(args)
	logger = Logger(args)
	for i in tqdm(range(len(data['examples'])),'Graph ID:'):
		G = data['examples'][i][0]
		grt = data['examples'][i][1]

		train, test = grt['train'], grt['test']
		train_inds = grt['train_idx']
		test_inds = grt['test_idx']


		# Run basline
		if args.algorithm == "jac":
			y_pred = jaccard(G, test)
			y_pred_train = jaccard(G, train)
		elif args.algorithm == "adar":
			y_pred = adamic_adar(G, test)
			y_pred_train = adamic_adar(G, train)
		elif args.algorithm == "prf":
			y_pred = preferential_attachment(G, test)
			y_pred_train = preferential_attachment(G, train)

		# Run greedy post processing 
		thresh = np.median(y_pred_train)
		y_pred_train = greedy_wrapper(G, data['communities'], train, y_pred_train, thresh, args.change_percent)
		y_pred = greedy_wrapper(G, data['communities'], test, y_pred, thresh, args.change_percent)

		# calculate accurcay of postprocessing 
		acc , roc_auc, _ = get_accuracy_scores(grt['test'], y_pred)
		logger.accs.append(acc)
		logger.aucs.append(roc_auc)

		# calculate the modred for postprocessing 
		modularity_new, modularity_ground = get_modularity(G,y_pred,data['communities'] , grt['test'])
		logger.modularities.append((modularity_ground- modularity_new)/np.abs(modularity_ground)) 
		logger.y_hats.append((y_pred_train,y_pred))       

	logger.averages()
	#logger.log_results()
	logger.print_results()
	#logger.pickle_results()


if __name__ =="__main__":
	main()