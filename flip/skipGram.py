import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center node.
    v_embedding: Embedding for neighbor nodes.
"""

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, sparse):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=sparse)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=sparse)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return self.u_embeddings.weight ,torch.mean(score + neg_score)
    def return_embedding(self):
        return self.u_embeddings.weight.data

    def save_embedding(self, id2node, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2node), self.emb_dimension))
            for wid, w in id2node.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(w), e))
