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


def get_args_from_dict(config):
	args = types.SimpleNamespace()	
	with open(config, 'r') as f:
		cfg_dict = json.load(f)
	for k,v in cfg_dict.items():
		if v == "True":
			v = True
		elif v == "False":
			v = False
		setattr(args, k, v)
	return args

def create_q2l_model(config):
	args = get_args_from_dict(config)
	print(args)
	model = build_q2l(args)
	model = model.cuda()	
	# Check why load clean doesnt work!
	# checkpoint = torch.load(args.resume, map_location="cpu")
	# model.load_state_dict(checkpoint["state_dict"])
	checkpoint = torch.load(args.resume, map_location='cpu')
	state_dict = clean_state_dict(checkpoint['state_dict'])
	model.load_state_dict(state_dict, strict=True)

	return model

