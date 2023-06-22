import os
import time
import random
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm.std import tqdm
from model import TKGQA
from transformers import AdamW
from torch.utils.data import DataLoader
from utils import (AverageMeter, RankingLoss, save_checkpoint)
from conv_data import ConvDataset
from constants import *

# create directories for experiments
logging_path = f'{str(ROOT_PATH)}/{args.logs}'
Path(logging_path).mkdir(parents=True, exist_ok=True)
checkpoint_path = f'{str(ROOT_PATH)}/{args.snapshots}'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{logging_path}/memory.log', 'w'),
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

def main():
    # log arguments
    logger.info(' '.join(f'{k}={v}' for k, v in vars(args).items()))
   
    # load ent2id and rel2id
    id2ent = load_item2id(ENTITY_PATH)
    id2rel =load_item2id(RELATION_PATH)
    # prepare training data
    logger.info('start training prepare')
    train_data = ConvDataset(TRAIN, id2ent, id2rel)
    logger.info('start val prepare')
    val_data = ConvDataset(VAL, id2ent, id2rel)

    model = TKGQA().to(DEVICE)

    # define framework loss
    criterion = RankingLoss

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # load checkpoint
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    # log num of params
    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # prepare training loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    logger.info('Training loader prepared.')

    # prepare validation loader
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    logger.info('Validation loader prepared.')
    
    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        # save the model
        if (epoch+1) % args.valfreq == 0 or (epoch+1) > args.min_val_epoch:
            val_losses = validate(val_loader, model, criterion)
            logger.info(f'Val Ranking loss: {val_losses.avg:.4f}')

            save_checkpoint({
                EPOCH: epoch + 1,
                STATE_DICT: model.state_dict(),
                OPTIMIZER: optimizer.state_dict(),
                VAL_LOSS: val_losses.avg},
                path=checkpoint_path)
                
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, batch in enumerate(train_loader):
        candi_path_indices = batch[FILTER_PATH_INICES].to(DEVICE)
        qu_emb = batch[QUESTION_EMB].to(DEVICE)
        path_index = batch[PATH_INDEX].to(DEVICE)
        ranking_target = batch[RANKING_TARGET].to(DEVICE)
        input = {
            FILTER_PATH_INICES: candi_path_indices,
            QUESTION_EMB: qu_emb,
            PATH_INDEX: path_index
        }

    
        target = ranking_target
        
       
        # compute output
        output = model(input)
        loss = criterion(output, target)
        
        # record loss
        losses.update(loss.data, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    logger.info(f'Epoch {epoch+1} - Ranking loss: {losses.avg:.4f}')
def validate(loader, model, criterion):
    
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch in loader:
        candi_path_indices = batch[FILTER_PATH_INICES].to(DEVICE)
        qu_emb = batch[QUESTION_EMB].to(DEVICE)
        path_index = batch[PATH_INDEX].to(DEVICE)
        ranking_target = batch[RANKING_TARGET].to(DEVICE)
        input = {
            FILTER_PATH_INICES: candi_path_indices,
            QUESTION_EMB: qu_emb,
            PATH_INDEX: path_index
        }
       
        target = ranking_target
        

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.data, args.batch_size)

    return losses

if __name__ == '__main__':
    main()