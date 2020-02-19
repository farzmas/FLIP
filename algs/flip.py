import random
import numpy as np
import networkx as nx
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt  
from torch.utils.data import DataLoader



from tqdm import tqdm, trange
from scipy import sparse

from skipGram.skipGram import SkipGramModel
from skipGram.data_reader import DataReader, node2vecDataset
from utils.eval import get_accuracy_scores, get_modularity


def xavier_init(m):
	""" Xavier initialization """
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0)

def to_sparse(x):
	""" converts dense tensor x to sparse format """
	x_typename = torch.typename(x).split('.')[-1]
	sparse_tensortype = getattr(torch.sparse, x_typename)

	indices = torch.nonzero(x)
	if len(indices.shape) == 0:  # if all elements are zeros
		return sparse_tensortype(*x.shape)
	indices = indices.t()
	values = x[tuple(indices[i] for i in range(indices.shape[0]))]
	return sparse_tensortype(indices, values, x.size())

class Discriminator(nn.Module):
	def __init__(self, args, n_com):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(2*args.dimensions,args.dimensions),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(args.dimensions,1),
			nn.Sigmoid(),
		)
		self.model.apply(xavier_init)
	def forward(self, embd , idx_1, idx_2):
		link_embd = torch.cat((embd[idx_1,:],embd[idx_2,:]), 1)
		y_hat = self.model(link_embd)
		return y_hat
	
	
class Link_prediction(nn.Module):
	def __init__(self, args):
		super(Link_prediction, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(2*args.dimensions,args.dimensions),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(args.dimensions,1),
			nn.Sigmoid(),
		)
		self.model.apply(xavier_init)
	def forward(self, embd , idx_1, idx_2):
		link_embd = torch.cat((embd[idx_1,:],embd[idx_2,:]), 1)
		y_hat = self.model(link_embd)
		return y_hat

class DW_GAN_LP(object):
	def __init__(self, args, G, communities, train_data ,f_id = '', logger = None):
		self.args = args 
		self.n = G.number_of_nodes() # number of nodes
		self.G = G # networkx graph 
		self.f_id = f_id #  
		self.file_name = f_id+ self.args.file_name.split('.')[0] +'.txt' # the generated walks file name
		self.data = DataReader(self.args.walk_path+self.file_name, 1)  # walks generated for the deepwalk 
		dataset = node2vecDataset(self.data, self.args.window_size)
		self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
									 shuffle=False, num_workers=0, collate_fn=dataset.collate)
		self.train_data = train_data # positive and negative examples for link prediction 
		self.train_init(communities)
		self.logger = logger
		self.Dis = Discriminator(self.args, len(communities))
		self.Gen = SkipGramModel(self.n,self.args.dimensions, args.sparse)
		self.LP = Link_prediction(self.args)
		if self.args.cuda:
			self.Gen = self.Gen.cuda()
			self.Dis = self.Dis.cuda()
			self.LP = self.LP.cuda()
			self.device = torch.device("cuda")
		else: 
			self.device = torch.device("cpu")
		if args.opt == 'adagrad':
			self.G_solver = opt.Adagrad(self.Gen.parameters(), lr= self.args.learning_rate)
			self.D_solver = opt.Adagrad(self.Dis.parameters(), lr= self.args.learning_rate)
			self.L_solver = opt.Adagrad(self.LP.parameters(), lr= self.args.learning_rate)
		elif args.opt == 'adam':
			self.G_solver = opt.Adam(self.Gen.parameters(), lr= self.args.learning_rate)
			self.D_solver = opt.Adam(self.Dis.parameters(), lr = self.args.learning_rate)
			self.L_solver = opt.Adam(self.LP.parameters(), lr = self.args.learning_rate)
		else:
			raise ValueError('optimizer must be either adagrad or adam')  
			
	def train(self, valid_data=None):
		lp_criterion = nn.BCELoss()
		d_criterion = nn.BCELoss()
		if self.args.cuda:
			self.labels_c = self.labels_c.cuda()
			self.labels_lp = self.labels_lp.cuda()
		if self.args.report_acc:
			self.logger.avg_losses_d[self.f_id] = list()
			self.logger.avg_losses_g[self.f_id] = list()
			self.logger.avg_losses_l[self.f_id] = list()
		self.epochs = trange(self.args.epochs, desc="Loss")
		for epoch in self.epochs:
			if self.args.report_acc:
				losses_d = list()
				losses_g = list()
				losses_l = list()
				valid_auc = list()
				valid_mod = list()
			for i, sample_batched in enumerate(tqdm(self.dataloader,)):
			   # ---------------------
				#  Train Discriminator
				# ---------------------
				# Feed forward in discriminator  for both communities 
				embd = self.Gen.return_embedding()    
				idx_start, idx_end = self.mini_batch(i)
				y_dis = self.Dis(embd, self.train_link_1[idx_start:idx_end], self.train_link_2[idx_start:idx_end])
				D_loss = d_criterion(y_dis, self.labels_c[idx_start:idx_end]) 
				# backward propagation
				self.D_solver.zero_grad()
				D_loss.backward()
				self.D_solver.step()
				# -----------------
				#  Train Link prediction
				# -----------------
				# Feed forward in link prediction
				y_lp = self.LP(embd, self.train_link_1[idx_start:idx_end], self.train_link_2[idx_start:idx_end])
				lp_loss = lp_criterion(y_lp, self.labels_lp[idx_start:idx_end])  
				
				# backward propagation for link prediction
				self.L_solver.zero_grad()
				lp_loss.backward()
				self.L_solver.step()
				
				#-----------------
				#  Train Generator
				# -----------------
				if len(sample_batched[0]) > 1:
					pos_u = sample_batched[0].to(self.device)
					pos_v = sample_batched[1].to(self.device)
					neg_v = sample_batched[2].to(self.device)
					embd, embd_loss = self.Gen(pos_u, pos_v, neg_v)  
					# Feed forward in discriminator  for both communities 
					y_dis = self.Dis(embd, self.train_link_1[idx_start:idx_end], self.train_link_2[idx_start:idx_end])
					D_loss = d_criterion(y_dis, self.labels_c[idx_start:idx_end]) 
					# Feed forward in link prediction
					y_lp = self.LP(embd, self.train_link_1[idx_start:idx_end], self.train_link_2[idx_start:idx_end])
					# Defining the loss for link prediction
					lp_loss = lp_criterion(y_lp, self.labels_lp[idx_start:idx_end])  
					G_loss = (1-self.args.beta_d)*embd_loss - self.args.beta_d*(D_loss) + self.args.beta_l*(lp_loss)
					#scheduler.step()
					self.G_solver.zero_grad()
					G_loss.backward()
					self.G_solver.step()
				self.epochs.set_description("( D-Loss=%g, G-Loss=%g,LP-Loss= %g)" % (round(D_loss.item(),4) , round(G_loss.item(),4),round(lp_loss.item(),4)))
				if self.args.report_acc:
					losses_g.append(round(G_loss.item(),4))
					losses_d.append(round(D_loss.item(),4))
					losses_l.append(round(lp_loss.item(),4))
			if self.args.report_acc:
				#y_pred_valid = self.test(valid_data)
				y_pred_train =self.test(self.train_data)
				acc , roc_auc, pa_ap = get_accuracy_scores(self.train_data, y_pred_train)
				print('train auc',round(roc_auc,4))
				#acc , roc_auc, pa_ap = get_accuracy_scores(valid_data, y_pred_valid)
				#print('valid auc',round(roc_auc,4))
				self.logger.avg_losses_d[self.f_id].append((round(np.mean(losses_d),4) , round(np.var(losses_d),4)))
				self.logger.avg_losses_g[self.f_id].append((round(np.mean(losses_g),4) , round(np.var(losses_g),4)))
				self.logger.avg_losses_l[self.f_id].append((round(np.mean(losses_l),4) , round(np.var(losses_l),4)))
		self.Gen.save_embedding(self.data.id2node,self.args.embedding_path+ self.file_name)
		if valid_data is None:
			return self.test(self.train_data)
		else:
			return self.test(self.train_data) , self.test(valid_data)

	def train_init(self,communities):
		#print('intializing the network')
		#self.labels_lp = torch.zeros(self.train_data.shape[0], 1)
		#self.labels_c = torch.zeros(self.train_data.shape[0], 1)
		self.labels_lp = list()
		self.labels_c = list()
		self.train_link_1 = list()
		self.train_link_2 = list()
		which_com = dict()
		G = self.G.copy()
		for c,community in enumerate(communities):
			for node in community:
				which_com[node] = c
		for i in range(self.train_data.shape[0]):
			row = self.train_data[i]
			self.labels_lp.append(int(row[2]))
			self.train_link_1.append(row[0])
			self.train_link_2.append(row[1])
			if which_com[row[0]]== which_com[row[1]]:
				self.labels_c.append(1)
			else:
				self.labels_c.append(0)
			G.add_edge(row[0],row[1])
		for edge in self.G.edges():
			self.labels_lp.append(1)
			self.train_link_1.append(edge[0])
			self.train_link_2.append(edge[1])
			if which_com[edge[0]]== which_com[edge[1]]:
				self.labels_c.append(1)
			else:
				self.labels_c.append(0)
		H = nx.complement(G)
		negative_examples = random.sample(H.edges(),len(self.G.edges()))
		for i in range(len(negative_examples)):
			pair = negative_examples[i]
			self.labels_c.append(0)
			self.train_link_1.append(pair[0])
			self.train_link_2.append(pair[1])
			if which_com[pair[0]]== which_com[pair[1]]:
				self.labels_c.append(1)
			else:
				self.labels_c.append(0)
		l = len(self.labels_lp)
		indices = np.arange(l)
		self.train_link_1 = torch.Tensor(self.train_link_1)[indices].long()
		self.train_link_2 = torch.Tensor(self.train_link_2)[indices].long()
		self.labels_lp = torch.Tensor(self.labels_lp)[indices].unsqueeze(1)
		self.labels_c = torch.Tensor(self.labels_c)[indices].unsqueeze(1)
		self.mb_size = l//len(self.dataloader)
	def mini_batch(self, idx):
		idx_start = self.mb_size*idx
		idx_end = idx_start  +self.mb_size
		return idx_start, idx_end
	
	def test(self, test_data):
		test_link1 = list()       
		test_link2 = list()
		for i in range(test_data.shape[0]):
			row = test_data[i]
			test_link1.append(row[0])
			test_link2.append(row[1])
		embd = self.Gen.return_embedding()
		y_hat = self.LP(embd, test_link1, test_link2)
		if self.args.cuda==1:
			return y_hat.cpu().detach().numpy()
		else:
			return y_hat.detach().numpy()
		








