import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


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


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, root, data_np, target_np, transform):
        self.root = root
        self.data_np = data_np
        self.target_np = target_np
        self.transform = transform

    def __getitem__(self, index):
        img_path, target = self.data_np[index], self.target_np[index]

        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        img = self.transform(img)

        return img, torch.from_numpy(target)

    def __len__(self):
        return len(self.data_np)

class TensorDatasetNW(Dataset):
    def __init__(self, data_np, target_np, transform):

        self.data_np = data_np
        self.target_np = target_np
        self.transform = transform

    def __getitem__(self, index):
        img_path, target = self.data_np[index], self.target_np[index]

        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, torch.from_numpy(target)

    def __len__(self):
        return len(self.data_np)

class TensorDatasetVOC(Dataset):
    def __init__(self, data_np, target_np, transform=None):

        self.data_np = data_np
        self.target_np = target_np
        self.transform = transform
        self.root = '/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/pascalvoc/VOCtrainval/'
        self.path_devkit = os.path.join(self.root, 'VOCdevkit')
        self.path_images = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'JPEGImages')

    def __getitem__(self, index):
        img_path, target = self.data_np[index], self.target_np[index]

        img = Image.open(os.path.join(self.path_images, img_path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # img = self.transform(img)

        return img, torch.from_numpy(target)

    def __len__(self):
        return len(self.data_np)


class CocoDetectionFiltered(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, label_indices_positive=None, label_indices_negative=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

        self.num_classes = 80
        self.train_data = []
        self.train_labels = []

        for index, _ in enumerate(self.ids):
            img_path, target = self.get_item_coco_numpy(index)
            # check if the specified labels are of the desired value

            target = target.astype(int)

            if label_indices_positive is not None:
                if int(sum(target[label_indices_positive])) != len(label_indices_positive):
                    continue

            if label_indices_negative is not None:                
                if sum(target[label_indices_negative]) != 0:
                    continue

            self.train_data.append(img_path)
            self.train_labels.append(target)

        num_samples = len(self.train_labels)
        self.num_samples = num_samples

        # Converting to numpy array
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)

    def __getitem__(self, index):
        # Don't forget train_data only keeps the path of the image
        # Loading the image will occur now (like in the original ASL repo)

        img_path, target = self.train_data[index], self.train_labels[index]

        # Here we load the image
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.from_numpy(target)


    def __len__(self):
        return self.num_samples


    # This method gives us numpy representation of img path and the corresponding label
    def get_item_coco_numpy(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = np.zeros(self.num_classes)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']

        return path, target


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
