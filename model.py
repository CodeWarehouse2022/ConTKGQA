import torch
import torch.nn as nn
import torch.nn.functional as F
from mem import MemN2N
from constants import *

# TKGQA
class TKGQA(nn.Module):
    def __init__(self):
        super(TKGQA, self).__init__()
        self.contrastive_module = ContrastiveModule()
        self.memory_module = MemN2N()
        self.kg_emb = nn.Embedding.from_pretrained(self.load_embedding(), padding_idx=args.pad_idx, freeze=True)
        
    def forward(self, input):
        paths_emb = self.kg_emb(input[FILTER_PATH_INICES])
        target_path_emb = self.kg_emb(input[PATH_INDEX]).squeeze(1)
        mem_out = self.memory_module(paths_emb, input[QUESTION_EMB])
        ranking_out = self.contrastive_module(mem_out,  target_path_emb)
        return ranking_out
        

    def load_embedding(self):
        kg_emb = torch.load(f'./datasets/{args.dataset}/kg/kg_embs.pt')
        zeros = torch.zeros(1, args.in_dim).long() # pad emb
        input_emb = torch.cat((kg_emb,zeros), 0)
        return input_emb
        
class LearnSequence(nn.Module):
    def __init__(self, in_dim=args.in_dim, emb_dim=args.emb_dim):
        super(LearnSequence, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))


class ContrastiveModule(nn.Module):
    def __init__(self):
        super(ContrastiveModule, self).__init__()

        #self.learn_conv_domain = LearnSequence(in_dim=args.emb_dim)
        self.learn_conv_domain = LearnSequence()
        self.learn_path        = LearnSequence()

    def forward(self, question, path):
        return {
            QUESTION: self.learn_conv_domain(question).squeeze(1),
            PATH: self.learn_path(path).squeeze(1)
        }

