import numpy as np
import torch
from torch.utils.data import Dataset




class DataReader:
	NEGATIVE_TABLE_SIZE = 1e8

	def __init__(self, inputFileName, min_count):

		self.negatives = []
		self.discards = []
		self.negpos = 0

		self.node2id = dict()
		self.id2node = dict()
		self.sentences_count = 0
		self.token_count = 0
		self.node_frequency = dict()

		self.inputFileName = inputFileName
		self.read_nodes(min_count)
		self.initTableNegatives()
		self.initTableDiscards()

	def read_nodes(self, min_count):
		node_frequency = dict()
		for line in open(self.inputFileName, encoding="utf8"):
			line = line.split()
			if len(line) > 1:
				line = [int(x) for x in line]
				self.sentences_count += 1
				for node in line:
					self.token_count += 1
					node_frequency[node] = node_frequency.get(node, 0) + 1
					# if self.token_count % 1000000 == 0:
					# 	#print("Read " + str(int(self.token_count / 1000000)) + "M nodes.")
		wid = 0
		for w in sorted(node_frequency.keys()):
			c = node_frequency[w]
			if c < min_count:
				continue
			self.node2id[w] = wid
			self.id2node[wid] = w
			self.node_frequency[wid] = c
			wid += 1
		#print("Total embeddings: " + str(len(self.node2id)))

	def initTableDiscards(self):
		t = 0.0001
		f = np.array(list(self.node_frequency.values())) / self.token_count
		self.discards = np.sqrt(t / f) + (t / f)

	def initTableNegatives(self):
		pow_frequency = np.array(list(self.node_frequency.values())) ** 0.5
		nodes_pow = sum(pow_frequency)
		ratio = pow_frequency / nodes_pow
		count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
		for wid, c in enumerate(count):
			self.negatives += [wid] * int(c)
		self.negatives = np.array(self.negatives)
		np.random.shuffle(self.negatives)

	def getNegatives(self, target, size): 
		response = self.negatives[self.negpos:self.negpos + size]
		self.negpos = (self.negpos + size) % len(self.negatives)
		if len(response) != size:
			return np.concatenate((response, self.negatives[0:self.negpos]))
		return response


# -----------------------------------------------------------------------------------------------------------------

class node2vecDataset(Dataset):
	def __init__(self, data, window_size):
		self.data = data
		self.window_size = window_size
		self.input_file = open(data.inputFileName, encoding="utf8")

	def __len__(self):
		return self.data.sentences_count

	def __getitem__(self, idx):
		while True:
			line = self.input_file.readline()
			if not line:
				self.input_file.seek(0, 0)
				line = self.input_file.readline()

			if len(line) > 1:
				
				nodes = line.split()
				nodes = [int(x) for x in nodes]
				if len(nodes) > 1:
					node_ids = [self.data.node2id[w] for w in nodes if
								w in self.data.node2id and np.random.rand() < self.data.discards[self.data.node2id[w]]]

					boundary = np.random.randint(1, self.window_size)
					return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(node_ids) for j, v in
							enumerate(node_ids[max(i - boundary, 0):i + boundary]) if u != v]

	@staticmethod
	def collate(batches):
		all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
		all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
		all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

		return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


	