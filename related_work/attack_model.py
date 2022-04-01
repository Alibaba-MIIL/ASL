# @Time      :2019/12/15 18:55
# @Author    :zhounan
# @FileName  :attack_model.py
from ml_cw_pytorch import MLCarliniWagnerL2
from ml_rank1_pytorch import MLRank1
from ml_rank2_pytorch import MLRank2
from ml_deepfool_pytorch import MLDeepFool
from mla_lp import MLLP
import numpy as np
import os
import math
from tqdm import tqdm
from datetime import datetime
import torch
tqdm.monitor_interval = 0

class AttackModel():
    def __init__(self, state):
        self.state = state
        self.y_target = state['y_target']
        self.y = state['y']
        self.data_loader = tqdm(state['data_loader'], desc='ADV')
        self.model = state['model']
        self.adv_save_x = state['adv_save_x']
        self.adv_batch_size = state['adv_batch_size']
        self.adv_begin_step = state['adv_begin_step']
        self.attack_model = None

    def attack(self):
        clip_min = 0.
        clip_max = 1.
        if self.state['adv_method'] == 'mla_lp':
            self.attack_model = MLLP(self.model)
            params = {'y_target': None,
                      'max_iter': 10,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.mla_lp(params)
        elif self.state['adv_method'] == 'ml_cw':
            self.attack_model = MLCarliniWagnerL2(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_cw(params)
        elif self.state['adv_method'] == 'ml_rank1':
            self.attack_model = MLRank1(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': 1.}
            self.ml_rank1(params)
        elif self.state['adv_method'] == 'ml_rank2':
            self.attack_model = MLRank2(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': 1.}
            self.ml_rank2(params)
        elif self.state['adv_method'] == 'ml_deepfool':
            self.attack_model = MLDeepFool(self.model)
            params = {'y_target': None,
                      'max_iter': 20,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_deepfool(params)
        else:
            print('please choose a correct adv method')

    def mla_lp(self, params):
        exec_times = []

        
        print(params)
        nsamples = 100
        for i, (x, target) in enumerate(self.data_loader):

            # Check sample limit
            if i >= nsamples:
                break

            # Keep track of time
            start_time = datetime.timestamp(datetime.now())

            params['batch_size'] = len(target)
            
            output = 2 * (torch.sigmoid(self.model(x.cuda())) > 0.5).int().detach().cpu().numpy() - 1
            t =  -output.copy()
         
            params['y_target'] = t

            # print(output)
            # print(params['y_target'])

            _, A_pos, A_neg, B_pos, B_neg = get_target_set(output, params['y_target'])
            A = A_pos + A_neg

            np.save(self.adv_save_x + 'mla_lp_' + 'clean{0}'.format(i), x)
            print(A)
            adv = self.attack_model.generate_np(x.cpu().numpy(), A, **params)
            np.save(self.adv_save_x + 'mla_lp_' + 'adv{0}'.format(i), adv)

            # check computation time
            end_time = datetime.timestamp(datetime.now())
            time_diff = end_time - start_time
            exec_times.append(time_diff)

        print(np.std(exec_times))
        print(np.mean(exec_times))


    def ml_cw(self, params):
        exec_times = []
        print(params)

        nsamples = 100
        for i, (x, target) in enumerate(self.data_loader):
            if i >= nsamples:
                break


            # TIME
            start_time = datetime.timestamp(datetime.now())

            params['batch_size'] = len(target)
            output = 2 * (torch.sigmoid(self.model(x.cuda())) > 0.5).int().detach().cpu().numpy() - 1
            params['y_target'] = -output

            np.save(self.adv_save_x + 'ml_cw_' + 'clean{0}'.format(i), x)

            adv = self.attack_model.generate_np(x.cpu().numpy(), **params)            

            # Save the adversarial
            np.save(self.adv_save_x + 'ml_cw2_' + 'adv{0}'.format(i), adv)

            # TIME
            end_time = datetime.timestamp(datetime.now())
            time_diff = end_time - start_time
            print(time_diff)
            exec_times.append(time_diff)

        print(np.std(exec_times))
        print(np.mean(exec_times))

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_rank1(self, params):
        y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        print(params)

        for i, (input, target) in enumerate(self.data_loader):
            print('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]
            params['y_tor'] = y_tor[begin:end]
            adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_rank2(self, params):
        y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        print(params)

        for i, (input, target) in enumerate(self.data_loader):
            print('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]
            params['y_tor'] = y_tor[begin:end]
            params['A_pos'] = A_pos[begin:end]
            params['A_neg'] = A_neg[begin:end]
            params['B_pos'] = B_pos[begin:end]
            params['B_neg'] = B_neg[begin:end]

            adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_deepfool(self, params):
        
        for i, (x, target) in enumerate(self.data_loader):
            

            # TIME
            start_time = datetime.timestamp(datetime.now())

            params['batch_size'] = len(target)
            output = 2 * (torch.sigmoid(self.model(x.cuda())) > 0.5).int().detach().cpu().numpy() - 1
            params['y_target'] = -output.copy()

            _, A_pos, A_neg, B_pos, B_neg = get_target_set(output, params['y_target'])
            A = 1 - (A_pos + A_neg)

            np.save(self.adv_save_x + 'ml_df_' + 'clean{0}'.format(i), x)

            adv = self.attack_model.generate_np(x.cpu().numpy(), A, **params)
            
            # Save the adversarial
            np.save(self.adv_save_x + 'ml_df_' + 'adv{0}'.format(i), adv)

            # TIME
            end_time = datetime.timestamp(datetime.now())
            time_diff = end_time - start_time
            print(time_diff)
            exec_times.append(time_diff)


def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_target_set(y, y_target):
    y[y == 0] = -1
    A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
    A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
    B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
    B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

    y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
    return y_tor, A_pos, A_neg, B_pos, B_neg
