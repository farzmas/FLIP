import torch
import random
import numpy as np
import networkx as nx
from tqdm import tqdm, trange
from scipy import sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt  
import pickle as pk
from  DeepWalk.skipGram import SkipGramModel
from DeepWalk.data_reader import DataReader, Word2vecDataset
from torch.utils.data import DataLoader
from linkprediction import link_predict, get_accuracy_scores, get_modularity
from sklearn.model_selection import train_test_split

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
	def __init__(self, args):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(args.dimensions,512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)
		self.model.apply(xavier_init)

	def forward(self, embd):
		#emd_flat = img.view(emd.shape[0], -1)
		validity = self.model(embd)
		return validity

# class Generator(nn.Module):
#     def __init__(self, args, shapes):
#         """
#         Setting up the layer.
#         :param args: Arguments object.
#         :param shapes: Shape of the target tensor.
#         """
#         super(Generator, self).__init__()
#         self.args = args
#         self.shapes = shapes
#         self.define_weights()
#         self.initialize_weights()    

class DW_GAN(object):
	def __init__(self, args, G, communities, grt = None ,f_id = '', logger = None):
		self.args = args 
		self.n = G.number_of_nodes()
		self.f_id = f_id
		self.file_name = f_id+ self.args.file_name.split('.')[0] +'.txt'
		if 'split' in self.file_name:
			self.file_name= self.file_name.split('_split')[0]+self.file_name.split('_split')[1]
		self.data = DataReader(self.args.walk_path+self.file_name, 1)
		dataset = Word2vecDataset(self.data, self.args.window_size)
		self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
									 shuffle=False, num_workers=0, collate_fn=dataset.collate)
		self.community_init( communities)
		self.grt = grt
		self.logger = logger
		if self.args.report_acc:
			self.G = G
			if 'split' in self.args.file_name:
				self.train, self.test,self.indx1,self.indx2 = self.grt['train'], self.grt['test'], self.grt['train_idx'], self.grt['test_idx']
			else:
				indices = np.arange(self.grt.shape[0])
				self.train_data , self.test_data,self.indx1,self.indx2= train_test_split(grt,indices, test_size= self.args.test_size , random_state=42)
			self.logger.init_report_training(self.f_id)
		self.Dis = Discriminator(self.args)
		self.Gen = SkipGramModel(self.n,self.args.dimensions, args.sparse)
		if self.args.cuda:
			self.Gen = self.Gen.cuda()
			self.Dis = self.Dis.cuda()
			self.device = torch.device("cuda")
		else: 
			self.device = torch.device("cpu")
		if args.opt == 'adagrad':
			self.G_solver = opt.Adagrad(self.Gen.parameters(), lr= self.args.learning_rate)
			self.D_solver = opt.Adagrad(self.Dis.parameters(), lr= self.args.learning_rate)
		elif args.opt == 'adam':
			#self.G_solver = opt.SparseAdam(self.Gen.parameters(), lr= self.args.learning_rate)
			#self.D_solver = opt.SparseAdam(self.Dis.parameters(), lr= self.args.learning_rate)
			self.G_solver = opt.Adam(self.Gen.parameters(), lr= self.args.learning_rate)
			self.D_solver = opt.Adam(self.Dis.parameters(), lr = self.args.learning_rate)
		else:
			raise ValueError('optimizer must be either adagrad or adam')   
	def train(self):
		n1 = len(self.com1[0]) # number of nodes in first community
		n2 = len(self.com2[0]) # number of ndoes in second community
		# Definig labels for each communites
		labels_1 = torch.zeros(n1, 1)
		labels_2 = torch.ones(n2, 1)
		if self.args.cuda:
			labels_1 = labels_1.cuda()
			labels_2 = labels_2.cuda()
		self.epochs = trange(self.args.epochs, desc="Loss")
		for epoch in self.epochs:
			if self.args.report_acc:
				losses_d = list()
				losses_g = list()
			for i, sample_batched in enumerate(tqdm(self.dataloader,)):
			   # ---------------------
				#  Train Discriminator
				# ---------------------
				# get the embding for Discriminator
				embd = self.Gen.return_embedding()

				# Feed forward in discriminator  for both communities 
				D_1 = self.Dis(embd[self.com1])
				D_2 = self.Dis(embd[self.com2])
				
				# Defining the loss for Discriminator
				D_2_loss = F.binary_cross_entropy(D_2, labels_2)
				D_1_loss = F.binary_cross_entropy(D_1, labels_1)
				D_loss = D_1_loss + D_2_loss        

				# backward propagation for discriminator
				self.D_solver.zero_grad()
				D_loss.backward()
				self.D_solver.step()
				
				# -----------------
				#  Train Generator
				# -----------------
				if len(sample_batched[0]) > 1:
					pos_u = sample_batched[0].to(self.device)
					pos_v = sample_batched[1].to(self.device)
					neg_v = sample_batched[2].to(self.device)
					embd, embd_loss = self.Gen(pos_u, pos_v, neg_v)  # Feed forward in discriminator  for both communities 
					
					#print(type(embd_loss), embd_loss.shape)
					D_1 = self.Dis(embd[self.com1])
					D_2 = self.Dis(embd[self.com2])

					D_2_loss = F.binary_cross_entropy(D_2, labels_2)
					D_1_loss = F.binary_cross_entropy(D_1, labels_1)
					#print(type(D_2_loss), D_2_loss.shape)
					G_loss = self.args.beta_g*embd_loss - self.args.beta_d*(D_1_loss + D_2_loss)
					
					#scheduler.step()
					self.G_solver.zero_grad()
					G_loss.backward()
					self.G_solver.step()
				self.epochs.set_description("( D-Loss=%g, G-Loss=%g)" % (round(D_loss.item(),4) , round(G_loss.item(),4)))
				if self.args.report_acc:
					losses_g.append(round(G_loss.item(),4))
					losses_d.append(round(D_loss.item(),4))
			if self.args.report_acc:
				self.logger.avg_losses_d[self.f_id].append((round(np.mean(losses_d),4) , round(np.var(losses_d),4)))
				self.logger.avg_losses_g[self.f_id].append((round(np.mean(losses_g),4) , round(np.var(losses_g),4)))
				embd = self.Gen.return_embedding().cpu().numpy()
				y_pred = link_predict(embd, self.train_data , self.test_data)
				acc , auc, _ = get_accuracy_scores(self.test_data, y_pred)
				self.logger.training_accs[self.f_id].append(round(acc,4))
				self.logger.training_aucs[self.f_id].append(round(auc,4))
				modularity_new, modularity_old = get_modularity(self.G, y_pred,self.communities , self.test_data)
				self.logger.training_modularities[self.f_id].append(round(((modularity_old- modularity_new)/modularity_old ),4))
				self.logger.print_training(self.f_id)

				#indx2_sorted =np.argsort(indx2)
		#return self.Gen.return_embedding()
		self.Gen.save_embedding(self.data.id2word,self.args.embedding_path+ self.file_name)
		return self.Gen.return_embedding().cpu().numpy()
	def community_init(self, communities):
		#n = self.G.number_of_nodes()
		idx1 = np.zeros(self.n)
		idx2 = np.zeros(self.n)
		for i in range(self.n):
			if i in communities[0]:
				idx1[i] = True
			if i in communities[1]:
				idx2[i] = True
		if self.args.report_acc:
			self.communities = communities
		self.com1 = np.where(idx1)
		self.com2 = np.where(idx2)
