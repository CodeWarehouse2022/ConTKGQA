import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='CronTKGQA')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--use_attn', action='store_true')
    # data
    parser.add_argument('--data_path', default='/data/final/')
    parser.add_argument('--dataset', default='CronQuestions')

    # experiments
    parser.add_argument('--logs', default='experiments/logs', type=str)
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--checkpoint', default='', type=str)

    # model
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--in_dim', default=384, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--pretrained_model', default='bert-base-uncased', type=str)

    # train
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--mismatch', default=0.8, type=int)
    parser.add_argument('--loss_ratio', default=0.99, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=5, type=int)
    parser.add_argument('--min_val_epoch', default=100, type=int)
    parser.add_argument('--resume', default='', type=str)

    return parser
