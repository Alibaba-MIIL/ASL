import os, sys
import os.path as osp
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

import csv
import tarfile
from urllib.parse import urlparse

import torch
import torch.utils.data as data

import pickle
# import util
# from util import *


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']



def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data

def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data

def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                
                for i in range(len(labels)):
                    # print(labels[i])
                    if labels[i] == -1.0:
                        labels[i] = 0.0
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


class Voc2007Classification(data.Dataset):
    def __init__(self, set, transform=None, target_transform=None, train=True, label_indices_positive=None, label_indices_negative=None):
        
        self.train = train

        
        if train == True:
            self.root = '../VOC2007/VOCtrainval'
        else:
            self.root = '../VOC2007/VOCtest'
        
        self.set = set
        self.path_devkit = os.path.join(self.root)
        self.path_images = os.path.join(self.root, 'VOCdevkit','VOC2007', 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = len(object_categories)
        seed = 1234
        # download dataset
        # download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        num_samples = len(self.images)
        # for index, _ in enumerate(self.ids):
        #     img_path, target = self.get_item_coco_numpy(index)
        #     # If we have multiple labels (we don't care about single-label samples)
        #     if np.count_nonzero(target) > 1:
        #         self.train_data.append(img_path)
        #         self.train_labels.append(target)
        
        # print("IMAGE:", self.images[51][1])
        # with open(inp_name, 'rb') as f:
        #     self.inp = pickle.load(f)
        # self.inp_name = inp_name

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))
        

        if self.train:

            self.train_data = []
            self.train_labels = []
            for i in range(len(self.images)):

                img_path = self.images[i][0]
                target = self.images[i][1]

                if label_indices_positive is not None:
                    if int(sum(target[label_indices_positive])) != len(label_indices_positive):
                        continue

                if label_indices_negative is not None:                
                    if int(sum(target[label_indices_negative])) != 0:
                        continue

                self.train_data.append(img_path)
                self.train_labels.append(target)

            self.train_data = np.array(self.train_data)
            self.train_labels = np.array(self.train_labels)

        else:

            self.test_data = []
            self.test_labels = []
            for i in range(len(self.images)):

                img_path = self.images[i][0]
                target = self.images[i][1]

                if label_indices_positive is not None:
                    if int(sum(target[label_indices_positive])) != len(label_indices_positive):
                        continue

                if label_indices_negative is not None:                
                    if sum(target[label_indices_negative]) != 0:
                        continue

                self.test_data.append(self.images[i][0])
                self.test_labels.append(self.images[i][1].cpu().numpy())

            self.test_data = np.array(self.test_data)
            self.test_labels = np.array(self.test_labels)
            

    def __getitem__(self, index):

        if self.train is True:
            path, target = self.train_data[index], self.train_labels[index]
        else:
            path, target = self.test_data[index], self.test_labels[index]
        # path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        # print(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return (img, path), target

    def __len__(self):
        if self.train is True:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
        # return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

class Voc2007ClassificationSingle(data.Dataset):
    def __init__(self, set, transform=None, target_transform=None, gold=False, gold_fraction=0.1, corruption_prob=0.4, corruption_type='unif',
                 shuffle_indices=None, train=True, distinguish_gold=True):        
        
        self.root = '/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/pascalvoc/VOCtrainval/'
        self.set = set
        self.path_devkit = os.path.join(self.root, 'VOCdevkit')
        self.path_images = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = len(object_categories)
        seed = 1234
        # download dataset
        # download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007', str(gold_fraction), corruption_type, str(corruption_prob), 'single')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        num_samples = len(self.images)
        
        # print("IMAGE:", self.images[51][1])
        # with open(inp_name, 'rb') as f:
        #     self.inp = pickle.load(f)
        # self.inp_name = inp_name

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))
        
        self.train_data = []
        self.train_labels = []
        for i in range(len(self.images)):
            if np.count_nonzero(self.images[i][1].cpu().numpy()) == 1:
                self.train_data.append(self.images[i][0])
                self.train_labels.append(self.images[i][1].cpu().numpy())

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
                

    def __getitem__(self, index):
        # path, target = self.images[index]
        path, target = self.train_data[index], self.train_labels[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
        # return (img, path), target

    def __len__(self):
        return len(self.train_data)

    def get_number_classes(self):
        return len(self.classes)


