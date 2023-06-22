import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import random
import logging
import numpy as np
from constants import *
import pandas as pd
from model import TKGQA
from conv_data import ConvDataset

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{str(ROOT_PATH)}/{args.logs}/test_memory_detailed.log', 'w'), #!!!
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def load_item2id(path):
    id2item = {}
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        idx, item = line.strip().split("\t")
        id2item[idx] = item
    return id2item


def log_ranking(similarities, tensor_gold_idx):
    prec1 = prec_at_1(similarities, tensor_gold_idx)
    hit5 = hits_at_k(similarities, tensor_gold_idx, 5)
    hit10 = hits_at_k(similarities, tensor_gold_idx, 10)


def model4_memory(test_data, model):
    for iter, conversation in enumerate(tqdm(test_data.expanded_data)):
        posnneg_paths = conversation[POSNNEG_PATHS]
        pos_paths = posnneg_paths[posnneg_paths[6]==1]
        gold_index = pos_paths.index.values.tolist()
    
        paths_emb = model.kg_emb(torch.LongTensor(posnneg_paths.index.values.tolist()).to(DEVICE))
        question_emb = torch.from_numpy(conversation[QUESTION_EMB]).to(DEVICE).reshape(1, -1)
        mem_out = model.memory_module(paths_emb, question_emb)
        qu_emb_con = model.contrastive_module.learn_conv_domain(mem_out).squeeze(0)

        learned_path_con = []
        valid_counts = []
        count = -1
        for index, path in posnneg_paths.iterrows():
            count += 1
            if index in gold_index:
                valid_counts.append(count)
            path_emb = model.kg_emb(torch.LongTensor([index]).to(DEVICE))
            path_emb_con = model.contrastive_module.learn_path(path_emb).squeeze(0) # 1xemb_dim
            learned_path_con.append(path_emb_con)
        learned_path_con = torch.stack(learned_path_con).squeeze(1) # path num x 768
        
        similarities = F.cosine_similarity(qu_emb_con, learned_path_con) # path num: 611
        tensor_gold_idx = torch.LongTensor([valid_counts]).to(DEVICE).squeeze(0)
        
        # hits@N
        if prec_at_1(similarities, tensor_gold_idx) > 0:
            metrics[HITS_AT_1].update(1)
        else:
            metrics[HITS_AT_1].update(0)
        if hits_at_k(similarities, tensor_gold_idx, k=5) > 0:
            metrics[HITS_AT_5].update(1)
        else:
            metrics[HITS_AT_5].update(0)
        if hits_at_k(similarities, tensor_gold_idx, k=10) > 0:
            metrics[HITS_AT_10].update(1)
        else:
            metrics[HITS_AT_10].update(0)
        
    #     if (iter+1) % 100 == 0:
    #         print_results(metrics)
    # print("==================")
    print_results(metrics)
    



def main():
    # prepare test data
    test_data = ConvDataset(TEST, id2ent, id2rel)
    
    # define model
    model = TKGQA().to(DEVICE)
    model_path = f'{ROOT_PATH}/{args.snapshots}/{args.checkpoint}'
    logger.info(f"=> loading checkpoint '{model_path}'")
    checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")
    # switch to evaluate mode
    model.eval()
    model4_memory(test_data, model)
    
    
def print_results(metrics):
    logger.info(f'Contrastive learning:')
    logger.info(f'\tHits@1: {metrics[HITS_AT_1].avg:.4f}')
    logger.info(f'\tHits@5: {metrics[HITS_AT_5].avg:.4f}')
    logger.info(f'\tHits@10: {metrics[HITS_AT_10].avg:.4f}')

id2ent = load_item2id(ENTITY_PATH)
id2rel =load_item2id(RELATION_PATH)
metrics = {
            HITS_AT_1: AverageMeter(),
            HITS_AT_5: AverageMeter(),
            HITS_AT_10: AverageMeter()
    }

if __name__ == '__main__':
    main()