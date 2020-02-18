import networkx as nx
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

class walk_generator():
	def __init__(self, G ,args):
		self.args = args
		self.size = G.number_of_nodes()
		self.normalize(G)
		
	def normalize(self, G):
		adj = nx.to_scipy_sparse_matrix(G)
		d = np.sum(np.array(adj.todense()),axis=0)
		adj_hat =sp.diags(1/d).dot(adj)
		self.adj = adj_hat

	def walker(self):
		nodes = list(range(self.size))
		np.random.shuffle(nodes)
		walks = list()
		for ct in tqdm(range(self.args.number_of_walks), desc ='number of walks'):
			for start_node in tqdm(nodes, desc ='start_node'):
				walk =[start_node]
				p = self.adj[start_node].todense() 
				p.resize((self.size,))
				for l in range(self.args.len_of_walks-1):
					node = np.random.choice(range(self.size),p= p)
					walk.append(node)
					p = self.adj[node].todense()
					p.resize((self.size,))
				walks.append(walk)
			np.random.shuffle(nodes)
		self.walks = walks

	def write_walks(self, file_id = str()):
		file_name = self.args.file_name.split('.')[0] +'.txt'
		file_ = open(self.args.walk_path+ file_id +file_name,'w')
		for walk in self.walks:
			line = str()
			for node in walk:
				line += str(node)+ ' '
			line += '\n'
			file_.write(line)
		file_.close()