import argparse  
	  
def parameter_parser():
	"""
	A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Chameleons dataset.
	The default hyperparameters give a good quality representation without grid search.
	Representations are sorted by node ID.
	"""

	parser = argparse.ArgumentParser(description = "Run Attention Walk.")

	parser.add_argument("--folder-path",
						nargs = "?",
						default =  "/data/Farzan/Fairness/graphs/",
					help = "path to input graph directory")
	parser.add_argument("--walk-path",
						nargs = "?",
						default =  "/data/Farzan/Fairness/walks/",
					help = "path to generated walks directory")
	parser.add_argument("--log-path",
						nargs = "?",
						default =  "/data/Farzan/Fairness/logs/",
					help = "path to input logs directory")
    
	parser.add_argument("--algorithm",
					nargs = "?",
					default = "n2v",
				help = "name of link prediciton algorithm to use: n2v for node2vec, dw for DeepWalk, jac for Jacard, adar for adamic_adar and  prf for preferential_attachment. Default is n2v.")

	parser.add_argument("--file-name",
						nargs = "?",
						default = "facebook_examples_rate=10.pk",
					help = "data file name")
	parser.add_argument("--log-file",
						nargs = "?",
						default = "base_log.txt",
					help = "log file name. Default is log.txt")
    
	parser.add_argument("--dimensions",
						type = int,
						default = 128,
					help = "Number of dimensions for representation learning methods . Default is 128.")

	parser.add_argument("--epochs",
						type = int,
						default = 1,
					help = "Number of gradient descent iterations for deep learning methods. Default is 1.")
	
	parser.add_argument("--workers",
						type = int,
						default = 1,
					help = "Number of workers for parallel execution. Default is 1.")

	parser.add_argument("--window-size",
						type = int,
						default = 10,
					help = "Skip-gram window size for walk base methods. Default is 10.")

	parser.add_argument("--len-of-walks",
						type = int,
						default = 80,
					help = "length of random walks for walk base methods. Default is 80.")

	parser.add_argument("--batch-size",
						type = int,
						default = 32,
					help = "minibach size for deep learning methods. Default is 32.")

	parser.add_argument("--single-id",
						type = int,
						default = 0,
					help = "running baselines on single graph with id single_id this is for single baseline version. Default is 0")

	parser.add_argument("--number-of-walks",
						type = int,
						default = 10,
					help = "number of random walks for walk base methods. Default is 10")

	parser.add_argument("--learning-rate",
						type = float,
						default = 0.001,
					help = "Gradient descent learning rate. Default is 0.001.")
	parser.add_argument("--p",
						type = float,
						default = 1,
					help = "node2vec hyperparameter p. Default is 1")

	parser.add_argument("--q",
						type = float,
						default = 1,
					help = "node2vec hyperparameter q. Default is 1")
	parser.add_argument("--cuda",
						type = bool,
						default = 0,
					help = "to use gpu set it 1. Default is 0")	
	parser.add_argument("--test-size",
					type = float,
					default = 0.95,
				help = "the proportion of the dataset to include in the test split. It should be between 0 and 1. Default is 0.5")
	parser.add_argument("--postproc",
					type = str,
					default = 'none',
				help = "whether or not to do equal odds post processing (eq_odds) or greedy (greedy). Default none.")
	parser.add_argument("--change_percent",
					type = float,
					default = 5,
				help = "Greedy post-processing hyper-parameter. Percent of predictions that can be changed to reduce modularity. Default 0.01.")
	parser.add_argument('--file', type=open, action=LoadFromFile)    
	return parser.parse_args()

class LoadFromFile (argparse.Action):
	def __call__ (self, parser, namespace, values, option_string = None):
		with values as f:
			parser.parse_args(f.read().split(), namespace)