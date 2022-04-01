#@Time      :2019/12/15 16:16
#@Author    :zhounan
#@FileName  :attack_main_pytorch.py
import sys
sys.path.append('../')
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.helper_functions.voc import Voc2007Classification
import argparse
import torch
import os
import numpy as np
import logging
from tqdm import tqdm
from attack_model import AttackModel
from create_q2l_model import create_q2l_model
from src.models import create_model
from src.helper_functions.helper_functions import CocoDetectionFiltered
from src.helper_functions.helper_functions import parse_args
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../../coco', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--dataset_type', default='MSCOCO_2014', type=str,
                    help='')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 32)')
parser.add_argument('--adv_batch_size', default=18, type=int,
                    metavar='N', help='batch size ml_cw, ml_rank1, ml_rank2 18, ml_lp 10, ml_deepfool is 10')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--adv_method', default='mla_lp', type=str, metavar='N',
                    help='attack method: ml_cw, ml_rank1, ml_rank2, ml_deepfool, ml_lp:97')
parser.add_argument('--adv_file_path', default='../data/voc2007/files/VOC2007/classification_mlgcn_adv.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/mlgcn/voc2007/', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=0, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 16)')


def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def init_log(log_file):
  new_folder(log_file)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

def get_target_label(y, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = y.copy()
    # o to -1
    y[y == 0] = -1
    if target_type == 'random_case':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            neg_idx = np.argwhere(y_i == -1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            neg_idx_c = np.random.choice(neg_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
            y[i, neg_idx_c] = -y[i, neg_idx_c]
    elif target_type == 'extreme_case':
        y = -y
    elif target_type == 'person_reduction':
        # person in 14 col
        y[:, 14] = -y[:, 14]
    elif target_type == 'sheep_augmentation':
        # sheep in 17 col
        y[:, 17] = -y[:, 17]
    elif target_type == 'hide_single':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
    return y

def gen_adv_file(model, target_type, adv_file_path):
    tqdm.monitor_interval = 0
    

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    # Pytorch Data loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    output = []
    # image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        nsamples = 100
        for i, (x, target) in enumerate(test_loader):
            if i >= nsamples:
                break
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            # image_name_list.extend(list(x[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        # image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx = []
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 2:
            true_idx.append(i)
    # adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type)
    y_target[y_target==0] = -1
    y[y==0] = -1

    # print(len(adv_image_name_list))
    # adv_labeled_data = {}
    # for i in range(len(adv_image_name_list)):
    #     adv_labeled_data[adv_image_name_list[i]] = y[i]
    # write_object_labels_csv(adv_file_path, adv_labeled_data)

    # save target y and ground-truth y to prepare attack
    # value is {-1,1}
    np.save('../adv_save/{0}/{1}/y_target.npy'.format('q2l',args.dataset_type), y_target)
    np.save('../adv_save/{0}/{1}/y.npy'.format('q2l',args.dataset_type), y)

# def evaluate_model(model):
#     tqdm.monitor_interval = 0

#     instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
#     data_path = '{0}/val2014'.format(args.data)

#     dataset = CocoDetectionFiltered(data_path,
#                                 instances_path,
#                                 transforms.Compose([
#                                     transforms.Resize((args.image_size, args.image_size)),
#                                     transforms.ToTensor(),
#                                     # normalize, # no need, toTensor does normalization
#                                 ]))

#     # Pytorch Data loader
#     test_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.workers, pin_memory=True)

#     output = []
#     y = []
#     test_loader = tqdm(test_loader, desc='Test')
#     with torch.no_grad():
#         for i, (input, target) in enumerate(test_loader):
#             x = input[0]
#             if use_gpu:
#                 x = x.cuda()
#             o = model(x).cpu().numpy()
#             output.extend(o)
#             y.extend(target.cpu().numpy())
#         output = np.asarray(output)
#         y = np.asarray(y)

#     pred = (output >= 0.5) + 0
#     y[y == -1] = 0

#     from utils import evaluate_metrics
#     metric = evaluate_metrics.evaluate(y, output, pred)
#     print(metric)

# def evaluate_adv(state):
#     model = state['model']
#     y_target = state['y_target']

#     adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
#     adv_file_list = os.listdir(adv_folder_path)
#     adv_file_list.sort(key=lambda x:int(x[16:-4]))
#     adv = []
#     for f in adv_file_list:
#         adv.extend(np.load(adv_folder_path+f))
#     adv = np.asarray(adv)
#     dl1 = torch.utils.data.DataLoader(adv,
#                                       batch_size=args.batch_size,
#                                       shuffle=False,
#                                       num_workers=args.workers)

#     data_transforms = transforms.Compose([
#         Warp(args.image_size),
#         transforms.ToTensor(),
#     ])
#     adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/voc2007/voc_glove_word2vec.pkl')
#     adv_dataset.transform = data_transforms
#     dl2 = torch.utils.data.DataLoader(adv_dataset,
#                                               batch_size=args.batch_size,
#                                               shuffle=False,
#                                               num_workers=args.workers)
#     dl2 = tqdm(dl2, desc='ADV')

#     adv_output = []
#     norm_1 = []
#     norm = []
#     max_r = []
#     mean_r = []
#     rmsd = []
#     with torch.no_grad():
#         for batch_adv_x, batch_test_x in zip(dl1, dl2):
#             if use_gpu:
#                 batch_adv_x = batch_adv_x.cuda()
#             adv_output.extend(model(batch_adv_x).cpu().numpy())
#             batch_adv_x = batch_adv_x.cpu().numpy()
#             batch_test_x = batch_test_x[0][0].cpu().numpy()

#             batch_r = (batch_adv_x - batch_test_x)
#             batch_r_255 = ((batch_adv_x / 2 + 0.5) * 255) - ((batch_test_x / 2 + 0.5) * 255)
#             batch_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
#             batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
#             norm.extend(batch_norm)
#             rmsd.extend(batch_rmsd)
#             norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
#             max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
#             mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
#     adv_output = np.asarray(adv_output)
#     adv_pred = adv_output.copy()
#     adv_pred[adv_pred >= (0.5+0)] = 1
#     adv_pred[adv_pred < (0.5+0)] = -1
#     print(adv_pred.shape)
#     print(y_target.shape)
#     adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
#     attack_fail_idx = np.argwhere(adv_pred_match_target==0).flatten().tolist()
#     attack_fail_idx = np.argwhere(adv_pred_match_target==0).flatten().tolist()

#     np.save('{}_attack_fail_idx.npy'.format(args.adv_method), attack_fail_idx)
#     norm = np.asarray(norm)
#     max_r = np.asarray(max_r)
#     mean_r = np.asarray(mean_r)
#     rmsd = np.asarray(rmsd)
#     norm = np.delete(norm, attack_fail_idx, axis=0)
#     max_r = np.delete(max_r, attack_fail_idx, axis=0)
#     norm_1 = np.delete(norm_1, attack_fail_idx, axis=0)
#     mean_r = np.delete(mean_r, attack_fail_idx, axis=0)
#     rmsd = np.delete(rmsd, attack_fail_idx, axis=0)

#     from utils import evaluate_metrics
#     metrics = dict()
#     y_target[y_target==-1] = 0
#     metrics['ranking_loss'] = evaluate_metrics.label_ranking_loss(y_target, adv_output)
#     metrics['average_precision'] = evaluate_metrics.label_ranking_average_precision_score(y_target, adv_output)
#     metrics['auc'] = evaluate_metrics.roc_auc_score(y_target, adv_output)
#     metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
#     metrics['norm'] = np.mean(norm)
#     metrics['norm_1'] = np.mean(norm_1)
#     metrics['rmsd'] = np.mean(rmsd)
#     metrics['max_r'] = np.mean(max_r)
#     metrics['mean_r'] = np.mean(mean_r)
#     print()
#     print(metrics)

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # set seed
    torch.manual_seed(123)
    if use_gpu:
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)


    # define dataset
    num_classes = 80

    # load torch model
    q2l = create_q2l_model('../config_coco.json')
    args.model_type = 'q2l'
    model = q2l

    if use_gpu:
        model = model.cuda()
   

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load target y and ground-truth y
    # value is {-1,1}
    

    state = {'model': model,
             'data_loader': data_loader,
             'adv_method': args.adv_method,
             'target_type': "extreme_case",
             'adv_batch_size': args.adv_batch_size,
             'y_target':0,
             'y': 0,
             'adv_save_x': '../adv_save/{0}/{1}/'.format('q2l',args.dataset_type),
             'adv_begin_step': args.adv_begin_step
             }

    # start attack
    attack_model = AttackModel(state)
    attack_model.attack()

    #evaluate_adv(state)
    #evaluate_model(model)

if __name__ == '__main__':
    main()