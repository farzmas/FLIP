from utils.parameter_parser import parameter_parser
from utils.utils import tab_printer, read_data,  walk_exist
from utils.eval import get_accuracy_scores, get_modularity
from algs.proximity_base import jaccard, adamic_adar, preferential_attachment
from algs.greedy import greedy_wrapper
from skipGram.walk_generator import walk_generator
from algs.flip import DW_GAN_LP
from utils.logger import Logger

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split




def main():
	args = parameter_parser()
	tab_printer(args)
	data = read_data(args)
	logger = Logger(args)
	train, test = train_test_split(data['examples'], test_size=args.test_size, random_state=1)
	G = data['G']

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

	# calculate accurcay and modred of baseline
	acc , roc_auc, _ = get_accuracy_scores(test , y_pred)
	modularity_new, modularity_ground  = get_modularity(G,y_pred,data['communities'] , test)
	modred = np.round((modularity_ground- modularity_new)/np.abs(modularity_ground), 4)
	print('baseline auc:', np.round(roc_auc,4))
	print('baseline modred:', modred)
	print()

	# Run greedy post processing 
	thresh = np.median(y_pred_train)
	y_pred_train = greedy_wrapper(G, data['communities'], train, y_pred_train, thresh, args.change_percent)
	y_pred = greedy_wrapper(G, data['communities'], test, y_pred, thresh, args.change_percent)

	# calculate accurcay and modred of postprocessing 
	acc , roc_auc, _ = get_accuracy_scores(test , y_pred)
	modularity_new, modularity_ground  = get_modularity(G,y_pred,data['communities'] , test)
	modred = np.round((modularity_ground- modularity_new)/np.abs(modularity_ground), 4)
	print('greedy postprocessing auc:', np.round(roc_auc,4))
	print('greedy postprocessing modred:', modred)
	print()

	if not walk_exist(args):
		walker = walk_generator(G,args)
		walker.walker()
		walker.write_walks()

	#train, valid = train_test_split(grt['train'], test_size = 0)
	DW = DW_GAN_LP(args, G , communities= data['communities'], train_data = train, logger = logger )
	y_pred_train = DW.train()
	# Get test results
	y_pred = DW.test(test)
	acc , roc_auc, pa_ap = get_accuracy_scores(test, y_pred, median = np.median(y_pred_train))
	modularity_new, modularity_ground  = get_modularity(G,y_pred,data['communities'] , test)
	modred = np.round((modularity_ground- modularity_new)/np.abs(modularity_ground), 4)
	print()
	print('flip auc:', np.round(roc_auc,4))
	print('flip modred:', modred)


if __name__ =="__main__":
	main()