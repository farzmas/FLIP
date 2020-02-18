from flip.parameter_parser import parameter_parser
from utils import tab_printer, read_data
from flip.walk_generator import walk_generator
from flip.utils import load_embd_from_txt, Logger, logger_exist, load_logger, walk_exist
from flip.flip import DW_GAN_LP
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from linkprediction import get_accuracy_scores, get_modularity

def main():	
	args = parameter_parser()
	tab_printer(args)
	logger = Logger(args)
	data = read_data(args)
	for i in tqdm(range(len(data['examples'])),'Graph ID:'):
		if not walk_exist(args, i):
			print('>>> starting random walk generation...')
			for ct, (G, _) in tqdm(enumerate(data['examples']),'Graph ID:'):
				walker = walk_generator(data['G'],args)
				walker.walker()
				walker.write_walks(str(ct))
			print('>>> generating random walks is over!')
		G = data['examples'][i][0]
		grt = data['examples'][i][1]
		#train, valid = train_test_split(grt['train'], test_size = 0)
		DW = DW_GAN_LP(args, G , communities= data['communities'], train_data = grt['train'], f_id =str(i), logger = logger )
		y_pred_train = DW.train()
		# Get test results
		y_pred = DW.test(grt['test'])
		acc , roc_auc, pa_ap = get_accuracy_scores(grt['test'], y_pred, median = np.median(y_pred_train))
		logger.accs.append(acc)
		logger.aucs.append(roc_auc)
		modularity_new, modularity_ground = get_modularity(G,y_pred,data['communities'] , grt['test'],median = np.median(y_pred_train))
		logger.modularities.append((modularity_ground- modularity_new)/np.abs(modularity_ground))
		logger.y_hats.append((y_pred_train,y_pred))
	logger.averages()
	#logger.log_results()
	logger.print_results()
	#logger.pickle_results()

if __name__ =="__main__":
    main()