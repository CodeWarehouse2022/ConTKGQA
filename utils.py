import re
import nltk
import torch
import sklearn
import torch.nn as nn
import numpy as np

# import constants
from constants import *


# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path):
    filename = f'{path}/epoch_{state["epoch"]}_loss_{state["val_loss"]:.4f}.pth.tar'
    torch.save(state, filename)

class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=args.margin).to(DEVICE)

    def forward(self, output, target, attn=None):
        return self.criterion(output[QUESTION], output[PATH], target)


def prec_at_1(predicted, actual):
    return len(np.intersect1d(predicted.topk(k=1)[1].cpu(), actual.cpu()))/actual.size(0)

def hits_at_k(predicted, actual, k=10):
    if k > len(predicted): k = len(predicted)
    return len(np.intersect1d(predicted.topk(k=k)[1].cpu(), actual.cpu()))/actual.size(0)
        