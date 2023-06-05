import logging
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import constants
from models import *
from data import prepare_datasets, load_embedding_weights, load_label_embedding, load_adj_matrix, \
    load_umls_embedding_weights, get_proc_diag_code
from trainer import train
import random
import os
import requests
import datetime
import shutil
import torch
import torch.nn as nn
import pandas as pd
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP


def get_hyper_params_combinations(args):
    params = OrderedDict(
        learning_rate_fine=args.learning_rate_fine,
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch
    )

    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def run(args, device, local_rank=None):
    train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer = prepare_datasets(args.data_setting,
                                                                                                   args.batch_size, args.max_len, args.bpe, args.nfold)
    logging.info(f'Taining labels are: {train_labels}\n')
    embed_weights = load_embedding_weights()

    label_desc = None
    label_desc, label_index_1, label_index_2 = load_label_embedding(train_labels, input_indexer.index_of(constants.PAD_SYMBOL), args.embed_size)
    adj_matrix = load_adj_matrix(args.data_setting, args.adj)
    model = None
    for hyper_params in get_hyper_params_combinations(args):

        if args.model == 'ours':

            model = Ours(embed_weights, args.embed_size, args.freeze_embed, args.max_len,
                                    args.num_trans_layers,args.num_attn_heads, args.trans_forward_expansion,
                                    train_set.get_code_count(), args.dropout_rate, label_desc, device, adj_matrix,
                                    args.batch_size, args.hidden_size, args.rnn_layer, label_index_1, label_index_2,
                                    args.make_sentence, use_gcn=True, lstm=True, pad_idx=0)

            model = torch.nn.DataParallel(model)
            model.to(device)
            logging.info(f"Training with: {hyper_params}")
            train(model.module, train_set, dev_set, test_set, hyper_params, args.batch_size, device, args.test, simi=False,
                  local_rank=local_rank)

        else:
            raise ValueError("Unknown value for args.model. Pick Transformer or TransICD")


if __name__ == "__main__":
    args = constants.get_args()
    if not os.path.exists('../results'):
        os.makedirs('../results')
    if not os.path.exists(f'../results/check_{args.name}'):
        os.makedirs(f'../results/check_{args.name}')
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=f'../results/check_{args.name}/app.log', filemode='w', format=FORMAT,
                        level=getattr(logging, args.log.upper()))
    logging.info(f'{args}\n')
    device = torch.device('cuda')
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    run(args, device)

