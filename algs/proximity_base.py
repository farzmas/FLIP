import networkx as nx
import numpy as np

def jaccard(G, grt):
	def predict(u, v):
		union_size = len(set(G[u]) | set(G[v]))
		if union_size == 0:
			return 0
		return len(list(nx.common_neighbors(G, u, v))) / union_size

	y_pred = list()
	for edge in grt[:,0:2]:
		y_pred.append(predict(edge[0], edge[1])) # predicted score       
	y_pred = np.array(y_pred)
	return y_pred

def adamic_adar(G, grt):
	def predict(u, v):
		return sum(1 / np.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
	y_pred = list()
	for edge in grt[:,0:2]:
		y_pred.append(predict(edge[0], edge[1])) # predicted score       
	y_pred = np.array(y_pred)
	return y_pred

def preferential_attachment(G, grt):
	def predict(u, v):
		return G.degree(u) * G.degree(v)
	y_pred = list()
	for edge in grt[:,0:2]:
		y_pred.append(predict(edge[0], edge[1])) # predicted score       
	y_pred = np.array(y_pred)
	return y_pred


