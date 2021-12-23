import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
import types
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import _init_paths
from dataset.get_dataset import get_datasets
from collections import namedtuple

from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict


def parser_args():
	available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384', 'Q2L-CvT_w24-384']

	parser = argparse.ArgumentParser(description='Query2Label for multilabel classification')
	parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14'])
	parser.add_argument('--dataset_dir', help='dir of dataset', default='../coco')
	
	parser.add_argument('--img_size', default=448, type=int,
						help='image size. default(448)')
	parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-TResL-448',
						choices=available_models,
						help='model architecture: ' +
							' | '.join(available_models) +
							' (default: Q2L-R101-448)')
	parser.add_argument('--config', default='config_new.json', type=str, help='config file')

	parser.add_argument('--output', metavar='DIR', 
						help='path to output folder')
	parser.add_argument('--loss', metavar='LOSS', default='asl', 
						choices=['asl'],
						help='loss functin')
	parser.add_argument('--num_class', default=80, type=int,
						help="Number of classes.")
	parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
						help='number of data loading workers (default: 8)')
	parser.add_argument('-b', '--batch-size', default=16, type=int,
						metavar='N',
						help='mini-batch size (default: 16), this is the total '
							'batch size of all GPUs')
	parser.add_argument('-p', '--print-freq', default=10, type=int,
						metavar='N', help='print frequency (default: 10)')
	parser.add_argument('--resume',default='checkpoint.pkl', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')

	parser.add_argument('--pretrained', dest='pretrained', action='store_true',
						help='use pre-trained model. default is False. ')

	parser.add_argument('--eps', default=1e-5, type=float,
					help='eps for focal loss (default: 1e-5)')

	# distribution training
	parser.add_argument('--world-size', default=-1, type=int,
						help='number of nodes for distributed training')
	parser.add_argument('--rank', default=-1, type=int,
						help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
						help='url used to set up distributed training')
	parser.add_argument('--seed', default=None, type=int,
						help='seed for initializing training. ')
	parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
	parser.add_argument('--amp', action='store_true',
						help='use mixture precision.')
	# data aug
	parser.add_argument('--orid_norm', action='store_true', default=False,
						help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')

	# * Transformer
	parser.add_argument('--enc_layers', default=1, type=int, 
						help="Number of encoding layers in the transformer")
	parser.add_argument('--dec_layers', default=2, type=int,
						help="Number of decoding layers in the transformer")
	parser.add_argument('--dim_feedforward', default=256, type=int,
						help="Intermediate size of the feedforward layers in the transformer blocks")
	parser.add_argument('--hidden_dim', default=128, type=int,
						help="Size of the embeddings (dimension of the transformer)")
	parser.add_argument('--dropout', default=0.1, type=float,
						help="Dropout applied in the transformer")
	parser.add_argument('--nheads', default=4, type=int,
						help="Number of attention heads inside the transformer's attentions")
	parser.add_argument('--pre_norm', action='store_true')
	parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
						help="Type of positional embedding to use on top of the image features")
	parser.add_argument('--backbone', default='resnet101', type=str,
						help="Name of the convolutional backbone to use")
	parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
						help='keep the other self attention modules in transformer decoders, which will be removed default.')
	parser.add_argument('--keep_first_self_attn_dec', action='store_true',
						help='keep the first self attention module in transformer decoders, which will be removed default.')
	parser.add_argument('--keep_input_proj', action='store_true', 
						help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
	args = parser.parse_args()

	args = {'dataname': 'coco14', 'dataset_dir': '../coco', 'img_size': 448, 'arch': 'Q2L-TResL-448', 'config': 'config_new.json', 'output': None, 'loss': 'asl', 'num_class': 80, 'workers': 8, 'batch_size': 16, 'print_freq': 10, 'resume': 'checkpoint.pkl', 'pretrained': False, 'eps': 1e-05, 'world_size': -1, 'rank': -1, 'dist_url': 'tcp://127.0.0.1:3451', 'seed': None, 'local_rank': None, 'amp': False, 'orid_norm': True, 'enc_layers': 1, 'dec_layers': 2, 'dim_feedforward': 2432, 'hidden_dim': 2432, 'dropout': 0.1, 'nheads': 4, 'pre_norm': False, 'position_embedding': 'sine', 'backbone': 'tresnetl', 'keep_other_self_attn_dec': False, 'keep_first_self_attn_dec': False, 'keep_input_proj': False, 'gamma_pos': 0.0, 'gamma_neg': 0.0, 'asl_clip': 0.0, 'weight_decay': 0.01, 'ema_decay': 0.9997, 'optim': 'Adam_twd'}
	# update parameters with pre-defined config file
	
	if args.config:
		with open(args.config, 'r') as f:
			cfg_dict = json.load(f)
		for k,v in cfg_dict.items():
			setattr(args, k, v)
	return args

def get_args_from_dict():
	args = types.SimpleNamespace()	
	with open('config_new.json', 'r') as f:
		cfg_dict = json.load(f)
	for k,v in cfg_dict.items():
		if v == "True":
			v = True
		elif v == "False":
			v = False
		setattr(args, k, v)

	return args


def get_args():
	args = parser_args()
	return args

def create_q2l_model():
	args = get_args_from_dict()
	model = build_q2l(args)
	model = model.cuda()	
	# Check why load clean doesnt work!
	checkpoint = torch.load(args.resume, map_location="cpu")
	model.load_state_dict(checkpoint["state_dict"])

	return model
