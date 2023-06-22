import re
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from constants import *
from tqdm import tqdm

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class ConvDataset(Dataset):
    def __init__(self, partition, id2ent, id2rel):
        if partition is None or partition not in [TRAIN, VAL, TEST]:
            raise Exception(f'Unknown partition type {partition}.')
        else:
            self.partition = partition
        self.max_path_size = {TRAIN: 9048, VAL: 1896, TEST: 600}[self.partition]
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.kg = pd.read_csv(f'./datasets/{args.dataset}/kg/full.txt', sep="\t", header=None)
        args.pad_idx = len(self.kg)
        self.all_index = range(len(self.kg))
        self.questions = self.load_pickle(os.path.join(QUESTION_PATH ,f'{partition.lower()}.pickle'))
        
        # load BERT
        self.path_emb_BERT = SentenceTransformer(SBERT_MODEL)

        self.mismatch = args.mismatch

        self.expanded_data = []
        for idx in tqdm(range(len(self.questions))):
            snap = self.questions[idx]
            answers = snap['answers']
            answer_type = snap['answer_type']
            relation = list(snap['relations'])[0]
            qu_type = snap['type']
            annotation = snap['annotation']
            template = snap['template']
            question_withQ = snap['question']     
            question = snap['paraphrases'][0]

            question_emb = self.path_emb_BERT.encode(question)
            all_paths = self.extract_path(annotation, template, question_withQ)
            filtered_paths = self.filter_path_ontime(qu_type, relation, answers, all_paths, annotation, answer_type, question_withQ, template)
            
            # delete neg neg path
            pos_paths = filtered_paths[filtered_paths[6]==1]
            pos_paths = pos_paths[~pos_paths.index.duplicated(keep='first')]
            neg_paths = filtered_paths[filtered_paths[6]==-1]
            neg_paths = neg_paths[~neg_paths.index.duplicated(keep='first')]
            pos_idx = pos_paths.index.values.tolist()
            neg_idx = neg_paths.index.values.tolist()
            inter_idx = set(pos_idx).intersection(set(neg_idx))
            neg_paths =  neg_paths.drop(list(inter_idx))
            filtered_paths = pd.concat([pos_paths, neg_paths])
            
            filtered_idx = filtered_paths.index.values.tolist() # paths with in the valied time contrains
            pad_filtered_idx = filtered_idx + ([args.pad_idx]* (self.max_path_size - len(filtered_idx)))
            self.expanded_data.append({
                ID: idx,
                POSNNEG_PATHS: filtered_paths,
                QUESTION_EMB: question_emb,
                FILTER_PATH_INICES: pad_filtered_idx
            })

            
    def __getitem__(self, index):
        question_data = self.expanded_data[index]
        
        # path
        posnneg_paths = question_data[POSNNEG_PATHS]
        # we force a mismatch given the probability
        match = np.random.uniform() > self.mismatch if self.partition == TRAIN else True
        target = match and 1 or -1
        if target == 1:
            # load one positive path
            path = posnneg_paths[posnneg_paths[6]==1].sample()
            
        else:
            if not any(posnneg_paths[6]==-1):
                path = posnneg_paths[posnneg_paths[6]==1].sample()
                target = 1
            else:
                # load one negative path
                path = posnneg_paths[posnneg_paths[6]==-1].sample()
        
        path_index = path.index.values[0]
        return{
            FILTER_PATH_INICES: torch.LongTensor(question_data[FILTER_PATH_INICES]), 
            QUESTION_EMB: question_data[QUESTION_EMB],
            PATH_INDEX: torch.LongTensor([path_index]),
            RANKING_TARGET: target
        }
         
    def __len__(self):
        return len(self.expanded_data)
    
    def load_pickle(self, path):
        file = open(path,'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file
    
    def extract_path(self, annotation, template, question):
        all_paths = pd.DataFrame()
        if 'head' in annotation:
            head_paths = self.kg[(self.kg[0]==annotation['head'])]
            head_paths[5] = 1
            all_paths = pd.concat([all_paths, head_paths]).drop_duplicates()
    
        if 'tail' in annotation:
            tail_paths = self.kg[(self.kg[2]==annotation['tail'])]
            tail_paths[5] = -1
            tail_paths=tail_paths.reindex(columns=[2,1,0,3,4,5])
            tail_paths.rename(columns={2: 0, 0: 2}, inplace=True)
            all_paths = pd.concat([all_paths, tail_paths]).drop_duplicates()

        if 'event_head' in annotation:
            eventhead_paths = self.kg[(self.kg[0]==annotation['event_head'])]
            eventhead_paths[5] = 1
            all_paths = pd.concat([all_paths, eventhead_paths]).drop_duplicates()
            eventhead_time_path = self.kg[(self.kg[0]==annotation['event_head']) & (self.kg[1]=='P793')]
            eventhead_time_path[5] = 1
            all_paths = pd.concat([all_paths, eventhead_time_path]).drop_duplicates()

        if '{tail2}' in template:
            template_spl = template.split()
            question_spl = question.split()
            assert len(template_spl) == len(question_spl)
            for spl_i in range(len(template_spl)):
                word = template_spl[spl_i]
                if word == "{tail2}":
                    tail2 = question_spl[spl_i]
                    break
            tail2_paths = self.kg[(self.kg[2]==tail2)]
            tail2_paths[5] = -1
            tail2_paths=tail2_paths.reindex(columns=[2,1,0,3,4,5])
            tail2_paths.rename(columns={2: 0, 0: 2}, inplace=True)
            all_paths = pd.concat([all_paths, tail2_paths]).drop_duplicates()
        all_paths = all_paths[all_paths[3] <= all_paths[4]]
        return all_paths
    
    def filter_path_ontime(self, qu_type, relation, answers, all_paths, annotation, answer_type, question, template):
        # 6: answer or not; 7: time relevant
        if qu_type == "simple_time":
            head = annotation['head']
            tail = annotation['tail']
            all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[3].isin(answers)) & (all_paths[4].isin(answers)) & (all_paths[5]==1), 1, -1)
            all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2]==head) & (all_paths[3].isin(answers)) & (all_paths[4].isin(answers)) & (all_paths[5]==-1), 1, all_paths[6])
            all_paths[7] = 1 # no time filter

        elif qu_type == "simple_entity":
            time = int(annotation['time'])
            if 'head' in annotation:
                head = annotation['head']
                all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[3] <= time) & (all_paths[4] >= time) & (all_paths[5]==1), 1, -1)
            else:
                tail = annotation['tail']
                all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[3] <= time) & (all_paths[4] >= time) & (all_paths[5]==-1), 1, -1)
            
            # based on time
            all_paths[7] = np.where((all_paths[3] <= time) & (all_paths[4] >= time), 1, -1)

        elif qu_type == "before_after":
            # label answer
            tail = annotation['tail']
            if 'head' in annotation:
                head = annotation['head']
                all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==1), 1, -1)
                all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==-1), 1, all_paths[6])                
                start_time = int(min(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail)& (all_paths[5]==1)][3]))
                end_time = int(max(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail)& (all_paths[5]==1)][4]))
            else:
                event_head = annotation['event_head']
                all_paths[6] = np.where((all_paths[0]==event_head) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==1), 1, -1)
                all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==-1), 1, all_paths[6])
                start_time = int(min(all_paths[(all_paths[0]==event_head) & (all_paths[1]=='P793')][3]))
                end_time = int(max(all_paths[(all_paths[0]==event_head) & (all_paths[1]=='P793')][4]))
            
            # label time relevant
            tag = annotation['type'] # before or after
            if tag == "before":
                all_paths[7] = np.where(all_paths[3] <= end_time, 1, -1)
            elif tag == "after":
                all_paths[7] = np.where(all_paths[4] >= start_time, 1, -1)

        elif qu_type == "first_last":
            adj = annotation['adj']
            if answer_type == "time":
                if ('head' in annotation) and ('tail' in annotation):
                    head = annotation['head']
                    tail = annotation['tail']
                    if adj == "first":
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[3].isin(answers)) & (all_paths[5]==1), 1, -1)
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2]==head) & (all_paths[3].isin(answers)) & (all_paths[5]==-1), 1, all_paths[6])
                    else:
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[4].isin(answers)) & (all_paths[5]==1), 1, -1)
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2]==head) & (all_paths[4].isin(answers)) & (all_paths[5]==-1), 1, all_paths[6])
                elif ('head' in annotation) and ('tail' not in annotation):
                    head = annotation['head']
                    if adj == "first":
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[3].isin(answers)) & (all_paths[5]==1), 1, -1)
                    else:
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[4].isin(answers)) & (all_paths[5]==1), 1, -1)
                else:
                    tail = annotation['tail']
                    if adj == "first":
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[3].isin(answers)) & (all_paths[5]==-1), 1, -1)
                    else:
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[4].isin(answers)) & (all_paths[5]==-1), 1, -1)
                all_paths[7] = 1
            else: #answer type is entity
                if 'head' in annotation:
                    head = annotation['head']
                    if adj == "first":
                        early_time = min(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[5]==1)][3].values)
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[3]==early_time) & (all_paths[5]==1), 1, -1)
                    else:
                        late_time = max(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[5]==1)][4].values)
                        all_paths[6] = np.where((all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[4]==late_time) & (all_paths[5]==1), 1, -1)

                else:
                    tail = annotation['tail']
                    if adj == "first":
                        early_time = min(all_paths[(all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[5]==-1)][3].values)
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[3]==early_time) & (all_paths[5]==-1), 1, -1)
                        #if (snap['entities'] == {'Q184299'}) and (snap['relations']=={'P39'}) and (answers == {'Q214559'}):
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) &  (all_paths[5]==-1), 1, -1)
                        #print(all_paths[3].count(r'\d'))
                        #exit(0)
                    else:
                        late_time = max(all_paths[(all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[5]==-1)][4].values)
                        all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[4]==late_time) & (all_paths[5]==-1), 1, -1)
                all_paths[7] = 1
        elif qu_type == "time_join":
            if '{tail2}' in template:
                head = annotation['head']
                tail = annotation['tail']
                template_spl = template.split()
                question_spl = question.split()
                assert len(template_spl) == len(question_spl)
                for spl_i in range(len(template_spl)):
                    word = template_spl[spl_i]
                    if word == "{tail2}":
                        tail2 = question_spl[spl_i]
                        break
                all_paths[6] = np.where((all_paths[0]==tail2) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==-1), 1, -1)
                start_time = int(min(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[5]==1)][3]))
                end_time = int(max(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[5]==1)][4]))
                # mark based on time
                all_paths[7] = np.where((all_paths[3]<=start_time) & (all_paths[4]>=start_time), 1, -1)
                all_paths[7] = np.where((all_paths[3]>=start_time) & (all_paths[3]<=end_time), 1, all_paths[7])
            else:
                tail = annotation['tail']
                if 'head' in annotation:
                    head = annotation['head']
                    all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==-1), 1, -1)
                    start_time = int(min(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[5]==1)][3]))
                    end_time = int(max(all_paths[(all_paths[0]==head) & (all_paths[1]==relation) & (all_paths[2]==tail) & (all_paths[5]==1)][4]))
                    # mark based on time
                    all_paths[7] = np.where((all_paths[3]<=start_time) & (all_paths[4]>=start_time), 1, -1)
                    all_paths[7] = np.where((all_paths[3]>=start_time) & (all_paths[3]<=end_time), 1, all_paths[7])
                else: # event head
                    event_head = annotation['event_head']
                    start_time = int(min(all_paths[(all_paths[0]==event_head) & (all_paths[1]=="P793") & (all_paths[5]==1)][3]))#.values[0]
                    end_time = int(max(all_paths[(all_paths[0]==event_head) & (all_paths[1]=="P793") & (all_paths[5]==1)][4]))#.values[0]
                    all_paths[6] = np.where((all_paths[0]==tail) & (all_paths[1]==relation) & (all_paths[2].isin(answers)) & (all_paths[5]==-1) & (((all_paths[3]>= start_time) & (all_paths[3]<=end_time)) | ((all_paths[3] <=start_time) & (all_paths[4] >= start_time))), 1, -1)
                    all_paths[7] = np.where((all_paths[3]<=start_time) & (all_paths[4]>=start_time), 1, -1)
                    all_paths[7] = np.where((all_paths[3]>=start_time) & (all_paths[3]<=end_time), 1, all_paths[7])
        return all_paths[all_paths[7]==1]


