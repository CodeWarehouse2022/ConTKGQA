import torch
import torch.nn as nn
from constants import *

class MemN2N(nn.Module):
    def __init__(self):
        super(MemN2N, self).__init__()
        self.max_hops = 2
        self.softmax = nn.Softmax()

    def forward(self, paths_emb, question_emb):
        u = list()
        u.append(question_emb)
        for hop in range(self.max_hops):
            u_i = u[-1].unsqueeze(1)
            p_i = self.softmax(torch.matmul(paths_emb, u_i.transpose(1, 2)))
            o_i = torch.sum(paths_emb*p_i, 1)
            u_iPlus1 = u[-1] + o_i
            u.append(u_iPlus1)
        return u[-1]

            