import numpy as np
import os
import pickle as pk
def load_embd_from_txt(args):
	embds = []
	for ct in range(10):
		file_name = args.file_name.split('.')[0] +'.txt'
		file_ = open(args.embedding_path+ str(ct)+ file_name, 'r')
		line = file_.readline().strip().split()
		embd = np.zeros((int(line[0]),int(line[1])))
		for line in file_:
			if len(line)>2:
				line = line.strip().split()
				node = int(line[0])
				line = [float(x) for x in line[1:]]
				embd[node,:] = np.array(line)
		embds.append(embd)
	return embds

class Logger():
	def __init__(self,args):
		self.args = args
		self.modularities = list()
		self.accs = list()
		self.aucs = list()
		self.valid_modularities = list()
		self.valid_accs = list()
		self.valid_aucs = list()
		self.avg_losses_d = dict()
		self.avg_losses_g = dict()
		self.avg_losses_l = dict()
		self.training_accs = dict()
		self.training_aucs = dict()
		self.training_modularities = dict()
		self.y_hats = list()

	def init_report_training(self,idx):
		self.avg_losses_d[idx] = list()
		self.avg_losses_g[idx] = list()
		self.avg_losses_l[idx] = list()
		self.training_accs[idx] = list()
		self.training_aucs[idx] = list()
		self.training_modularities[idx] = list()

	def averages(self):
		if len(self.valid_accs)>0:
			self.avg_valid_modularity =(np.round(np.mean(self.valid_modularities),decimals=4),
							  np.round(np.var(self.valid_modularities),decimals=4))
			self.avg_valid_acc = (np.round(np.mean(self.valid_accs),decimals=4), np.round(np.var(self.valid_accs),decimals=4))
			self.avg_valid_auc = (np.round(np.mean(self.valid_aucs),decimals=4), np.round(np.var(self.valid_aucs),decimals=4))
		
		self.avg_modularity =(np.round(np.mean(self.modularities),decimals=4),
							  np.round(np.var(self.modularities),decimals=4))
		self.avg_acc = (np.round(np.mean(self.accs),decimals=4), np.round(np.var(self.accs),decimals=4))
		self.avg_auc = (np.round(np.mean(self.aucs),decimals=4), np.round(np.var(self.aucs),decimals=4))

	def log_results(self):
		file = open(self.args.log_path + self.args.log_file,'w')
		file.write('modularity_reduction '+str(self.avg_modularity[0]) + ' +/- '+ str(self.avg_modularity[1])+'\n')
		file.write('acc '+str(self.avg_acc[0]) + ' +/- '+ str(self.avg_acc[1])+'\n')
		file.write('auc '+str(self.avg_auc[0])+ ' +/- '+ str(self.avg_auc[1])+'\n')
		file.close()
	def print_results(self):
		print('average modularity reduction = ',self.avg_modularity[0] , '+/-', self.avg_modularity[1])
		print('average acc = ', self.avg_acc[0], '+/-', self.avg_acc[1])
		print('average auc = ', self.avg_auc[0], '+/-', self.avg_auc[1])

	def pickle_results(self):
		pickle_file = self.args.log_file.split('.')[0]+'.pk'
		file = open(self.args.log_path + pickle_file,'wb')
		pk.dump(self, file )
		file.close()

	def print_training(self, idx):
		print('    average loss Generator =', self.avg_losses_g[idx][-1][0] ,'+/-',self.avg_losses_g[idx][-1][1] )
		print('    average loss Discriminator =', self.avg_losses_d[idx][-1][0] ,'+/-',self.avg_losses_d[idx][-1][1] )
		print('    link prediction acc =', self.training_accs[idx][-1] )
		print('    link prediction auc =', self.training_aucs[idx][-1] )
		print('    link prediction modularity reduction =', self.training_modularities[idx][-1] )

def logger_exist(args):
	path = args.log_path +args.log_file.split('.')[0]+'.pk'
	return os.path.isfile(path)

def walk_exist(args, i):
	# only check one incident 
	path = args.walk_path+ str(i) +args.file_name.split('.')[0] +'.txt'
	#print(path)
	return os.path.isfile(path)

def load_logger(args):
	path = args.log_path +args.log_file.split('.')[0]+'.pk'
	file = open(path,'rb')
	logger = pk.load(file)
	return logger