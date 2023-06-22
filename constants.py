import os
import torch
from pathlib import Path
from args import get_parser

# read parser
parser = get_parser()
args = parser.parse_args()

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))
QUESTION_PATH = f'./datasets/{args.dataset}/questions'
# model name
MODEL_NAME = 'CronTKGQA'

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
#torch.cuda.set_device(args.cuda_device)

# partition
TRAIN = 'train'
VAL = 'valid'
TEST = 'test'

# training
TASK = 'task'
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'
LOSS = 'loss'
VAL_LOSS = 'val_loss'

# testing
HITS_AT_1 = 'hits@1'
HITS_AT_5 = 'hits@5'
HITS_AT_10 = 'hits@10'
MR = 'mr'
MRR = 'mrr'
ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1_SCORE = 'f1_score'

# other
ENTITY_PATH = f'./datasets/{args.dataset}/kg/wd_id2entity_text.txt'
RELATION_PATH = f'./datasets/{args.dataset}/kg/wd_id2relation_text.txt'
ID = 'id'

PATH = 'path'
BERT_BASE_UNCASED = 'bert-base-uncased'
SBERT_MODEL = 'all-MiniLM-L6-v2'
RANKING_TARGET = 'ranking_target'
FILTER_PATH_INICES = 'filtered_path_indices'
PATH_INDEX = 'path_index'
QUESTION_EMB = 'question_emb'
PATHS_EMBS = 'paths_embs'
POSNNEG_PATHS = 'posnneg_paths'
PATH_EMB = 'path_emb'
QUESTION = 'question'
LOGITS = 'logits'
ENCODER_OUT = 'encoder_out'