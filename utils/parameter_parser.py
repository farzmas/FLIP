import argparse  
	
def parameter_parser():
	"""
	
	"""

	parser = argparse.ArgumentParser(description = "Run fair gan.")
	
	parser.add_argument("--folder-path",
						nargs = "?",
						default = "./data/graphs/",
					help = "path to input graph directory")


	parser.add_argument("--file-name",
						nargs = "?",
						default = "caseStudy.pk",
					help = "data file name")

	parser.add_argument("--walk-path",
						nargs = "?",
						default = "./data/walks/",
					help = "path to generated walk folder")

	parser.add_argument("--embedding-path", nargs = "?", 
						default = "./data/embds/",
						help= "path to generated embedding folder")

	parser.add_argument("--log-path",
					nargs = "?",
					default =  "./data/logs/",
				help = "path to input logs directory")

	parser.add_argument("--dimensions",
						type = int,
						default = 128,
					help = "Number of dimensions. Default is 128.")

	parser.add_argument("--epochs",
						type = int,
						default = 3,
					help = "Number of gradient descent iterations. Default is 3.")

	parser.add_argument("--window-size",
						type = int,
						default = 10,
					help = "Skip-gram window size. Default is 10.")

	parser.add_argument("--len-of-walks",
						type = int,
						default = 80,
					help = "length of random walks. Default is 80.")

	parser.add_argument("--batch-size",
						type = int,
						default = 32,
					help = "minibach size. Default is 32.")

	parser.add_argument("--test-size",
						type = float,
						default = 0.8,
					help = "link prediction test size. Default is 0.8 .")

	parser.add_argument("--number-of-walks",
						type = int,
						default = 10,
					help = "number of random walks. Default is 10")

	parser.add_argument("--learning-rate",
						type = float,
						default = 0.001,
					help = "Gradient descent learning rate. Default is 0.001.")
	parser.add_argument("--beta-g",
						type = float,
						default = 0.9,
					help = "generator hyperparameter. Default is 0.9")

	parser.add_argument("--beta-d",
						type = float,
						default = 0.1,
					help = "discriminator hyperparameter. Default is 0.1")
	parser.add_argument("--beta-l",
						type = float,
						default = 0.2,
					help = "link prediction hyperparameter for integerated version. Default is 0.5")
	parser.add_argument("--cuda",
						type = bool,
						default = 0,
					help = "to use gpu set it 1. Default is 0")
	parser.add_argument("--report-acc",
						type = bool,
						default = 0,
					help = "report acc during embedding training. Default is 0")
	parser.add_argument("--opt",
						nargs = "?",
						default = 'adam',
					help = "choose optimization algorithm. Can be 'adam' or 'adagrad'. Default is adam")
	parser.add_argument("--sparse",
						type = bool,
						default = False,
					help = "choose whether to use sparse or dense tensors. Default is dense")
	parser.add_argument("--mini-batchs-lp",
						type = bool,
						default = False,
					help = "choose whether to use sparse or dense tensors. Default is dense")

	parser.add_argument("--log-file",
						nargs = "?",
						default = "log.txt",
					help = "log file name. Default is log.txt")

	parser.add_argument("--algorithm",
					nargs = "?",
					default = "adar",
				help = "name of link prediciton baseline algorithm to use: jac for Jacard, adar for adamic_adar and  prf for preferential_attachment. Default is n2v.")

	parser.add_argument("--change_percent",
					type = float,
					default = 0.03,
				help = "Greedy post-processing hyper-parameter. Percent of predictions that can be changed to reduce modularity. Default 0.03.")

	parser.add_argument('--file', type=open, action=LoadFromFile)

	return parser.parse_args()

class LoadFromFile (argparse.Action):
	def __call__ (self, parser, namespace, values, option_string = None):
		with values as f:
			parser.parse_args(f.read().split(), namespace)