import os, sys
import os.path as osp
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


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


class NusWideFiltered(Dataset):
    def __init__(self, split, transform=None, label_indices_positive=None, label_indices_negative=None):
        img_dir='../NUS_WIDE/nus_wide'
        csv_path='../NUS_WIDE/nus_wid_data.csv'
        self.img_dir = img_dir
        self.csv_path = csv_path
        assert split in ['all', 'train', 'val']
        self.transform = transform
        self.num_classes = 81
        self.split = split

        if self.split == 'train':
            self.train_data = []
            self.train_labels = []
            self.data = self.preprocess()
            for idx, _ in enumerate(self.data):
                imgpath, labels = self.data[idx]

                if label_indices_positive is not None:
                    if int(sum(labels[label_indices_positive])) != len(label_indices_positive):
                        continue

                if label_indices_negative is not None:                
                    if int(sum(labels[label_indices_negative])) != 0:
                        continue
                
                self.train_data.append(imgpath)
                self.train_labels.append(labels)


            num_samples = len(self.train_labels)
            self.num_samples = num_samples
            self.train_data = np.array(self.train_data)
            self.train_labels = np.array(self.train_labels)

        else:

                self.test_data = []
                self.test_labels = []
                self.data = self.preprocess()
                for idx, _ in enumerate(self.data):
                    imgpath, labels = self.data[idx]
                    if np.count_nonzero(labels) > 1:
                        self.test_data.append(imgpath)
                        self.test_labels.append(labels)

                num_samples = len(self.test_labels)
                self.num_samples = num_samples

                # Converting to numpy array
                self.test_data = np.array(self.test_data)
                self.test_labels = np.array(self.test_labels)

    def preprocess(self):
        # read csv file
        df = pd.read_csv(self.csv_path)
        labels_col = df['label']
        labels_list_all = []
        for item in labels_col:
            i_labellist = str_to_list(item)
            labels_list_all.extend(i_labellist)
        labels_list_all = sorted(list(set(labels_list_all)))
        labels_map = {labelname:idx for idx, labelname in enumerate(labels_list_all)}
        length = len(labels_list_all)

        # generate itemlist
        res = []
        for index, row in df.iterrows():
            split_name = row[2]
            if split_name != self.split and self.split != 'all':
                continue
            filename = row[0]
            imgpath = osp.join(self.img_dir, filename)
            label = [labels_map[i] for i in str_to_list(row[1])]
            label_np = np.zeros(length, dtype='float32')
            for idd in label:
                label_np[idd] = 1.0
            res.append((imgpath, label_np))

        return res

    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.test_labels)
        #return self.num_samples

    def __getitem__(self, index: int):

        if self.split == 'train':
            img_path, target = self.train_data[index], self.train_labels[index]
        else:
            img_path, target = self.test_data[index], self.test_labels[index]

        #imgpath, labels = self.itemlist[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class NusWideAslDataset(Dataset):
    def __init__(self, split, gold=False, gold_fraction=0.1, 
                corruption_prob=0.0, corruption_type='unif', distinguish_gold=True, shuffle_indices=None, transform=None, seed=1):
        img_dir='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/dataset/'
        csv_path='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/nus_wid_data.csv'
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.split = split
        self.gold = gold
        self.gold_fraction = gold_fraction
        self.corruption_prob = corruption_prob
        assert split in ['all', 'train', 'val']
        self.transform = transform
        self.num_classes = 81

        #self.itemlist = self.preprocess()
        if self.split == 'train':
            self.train_data = []
            self.train_labels = []
            self.data = self.preprocess()
            for idx, _ in enumerate(self.data):
                imgpath, labels = self.data[idx]
                if np.count_nonzero(labels) > 1:
                    self.train_data.append(imgpath)
                    self.train_labels.append(labels)
            num_samples = len(self.train_labels)
            self.num_samples = num_samples

            self.train_data = np.array(self.train_data)
            self.train_labels = np.array(self.train_labels)

            if gold is True:
                if shuffle_indices is None:
                    avg_labels_per_image = 2.9  # From ASL paper
                    indices = np.arange(num_samples)

                    np.random.seed(seed)
                    np.random.shuffle(indices)

                    shuffled_train_labels = self.train_labels[indices]
                    # For the most cases it won't even go once
                    while np.count_nonzero(shuffled_train_labels[:int(gold_fraction * num_samples)].sum(axis=0)
                                           > avg_labels_per_image) < self.num_classes:
                        np.random.shuffle(indices)
                        shuffled_train_labels = self.train_labels[indices]
                else:
                    indices = shuffle_indices
                self.train_data = self.train_data[indices][:int(gold_fraction * num_samples)]
                if distinguish_gold:
                    self.train_labels = self.train_labels[indices][:int(gold_fraction * num_samples)] + self.num_classes
                else:
                    self.train_labels = self.train_labels[indices][:int(gold_fraction * num_samples)]
                self.shuffle_indices = indices
            else:
                indices = np.arange(len(self.train_data)) if shuffle_indices is None else shuffle_indices
                self.train_data = self.train_data[indices][int(gold_fraction * num_samples):]
                self.train_labels = self.train_labels[indices][int(gold_fraction * num_samples):]

                if corruption_type == 'flip':
                    raise Exception('Corruption type "flip" not implemeneted')
                    # C = flip_labels_C(self.corruption_prob, self.num_classes)
                    # self.C = C
                elif corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, self.num_classes)
                    self.C = C

                else:
                    assert False, "Invalid corruption type '{}' given. " \
                                  "Must be in ['unif', 'flip']".format(corruption_type)
                self.C_true = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
                np.random.seed(seed)
                tmp = 0
                if corruption_type == 'unif':
                    for i in range(len(self.train_labels)):
                        true_labels = np.nonzero(self.train_labels[i])[0]
                        for label in range(len(self.train_labels[i])):
                            if self.train_labels[i][label] == 1:
                                new_label = np.random.choice(self.num_classes, p=self.C[label])
                                # If the choice has been to corrupt this label
                                if new_label != label:
                                    if label == 1:
                                        tmp += 1
                                    # This ensures we are generating WRONG labels (not missing/weak labels)
                                    while self.train_labels[i][new_label] == 1 or new_label in true_labels:
                                        new_label = np.random.choice(self.num_classes, p=self.C[label])
                                    self.train_labels[i][label] = 0
                                    self.train_labels[i][new_label] = 1
                                    self.C_true[label][new_label] += 1
                                else:
                                    self.C_true[label][label] += 1
                    self.corruption_matrix = C
                    self.C_true /= np.sum(self.C_true, axis=1)
                elif corruption_type == 'flip':
                    raise RuntimeError("Not yet implemented")
                else:
                    raise RuntimeError("Not yet implemented")
        else:

            self.test_data = []
            self.test_labels = []
            self.data = self.preprocess()
            for idx, _ in enumerate(self.data):
                imgpath, labels = self.data[idx]
                if np.count_nonzero(labels) > 1:
                    self.test_data.append(imgpath)
                    self.test_labels.append(labels)

            num_samples = len(self.test_labels)
            self.num_samples = num_samples

            # Converting to numpy array
            self.test_data = np.array(self.test_data)
            self.test_labels = np.array(self.test_labels)




    def preprocess(self):
        # read csv file
        df = pd.read_csv(self.csv_path)
        labels_col = df['label']
        labels_list_all = []
        for item in labels_col:
            i_labellist = str_to_list(item)
            labels_list_all.extend(i_labellist)
        labels_list_all = sorted(list(set(labels_list_all)))
        labels_map = {labelname:idx for idx, labelname in enumerate(labels_list_all)}
        length = len(labels_list_all)

        # generate itemlist
        res = []
        for index, row in df.iterrows():
            split_name = row[2]
            if split_name != self.split and self.split != 'all':
                continue
            filename = row[0]
            imgpath = osp.join(self.img_dir, filename)
            label = [labels_map[i] for i in str_to_list(row[1])]
            label_np = np.zeros(length, dtype='float32')
            for idd in label:
                label_np[idd] = 1.0
            res.append((imgpath, label_np))

        return res

    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.test_labels)
        #return self.num_samples

    def __getitem__(self, index: int):

        if self.split == 'train':
            img_path, target = self.train_data[index], self.train_labels[index]
        else:
            img_path, target = self.test_data[index], self.test_labels[index]

        #imgpath, labels = self.itemlist[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def str_to_list(text):
    """
    input: "['clouds', 'sky']" (str)
    output: ['clouds', 'sky'] (list)
    """
    # res = []
    res = [i.strip('[]\'\"\n ') for i in text.split(',')]
    return res


# if __name__ == '__main__':
#     ds = NusWideAslDataset(
#             img_dir='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/dataset/images',
#             csv_path='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/nus_wid_data.csv',
#             split='train',
#             transform=None
#         )
#     print('len(ds):', len(ds))
#     # print(ds[0])

#     ds = NusWideAslDataset(
#             img_dir='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/dataset/images',
#             csv_path='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/nus_wid_data.csv',
#             split='val',
#             transform=None
#         )
#     print('len(ds):', len(ds))
#     print(ds[0])

class NusWideAslDatasetSingle(Dataset):
    def __init__(self, transform=None, seed=1):
        img_dir='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/dataset/'
        csv_path='/media/masoud/DATA/masoud_data/nus_wide/robust_multi_label-main/nus_wid_data.csv'
        self.img_dir = img_dir
        self.csv_path = csv_path
        
        self.transform = transform
        self.num_classes = 81

        self.train_data = []
        self.train_labels = []
        self.data = self.preprocess()
        for idx, _ in enumerate(self.data):
            imgpath, labels = self.data[idx]
            if np.count_nonzero(labels) == 1:
                #print("Single-Labels!")
                self.train_data.append(imgpath)
                self.train_labels.append(labels)
        num_samples = len(self.train_labels)
        self.num_samples = num_samples
        

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
        

    def preprocess(self):
        # read csv file
        df = pd.read_csv(self.csv_path)
        labels_col = df['label']
        labels_list_all = []
        for item in labels_col:
            i_labellist = str_to_list(item)
            labels_list_all.extend(i_labellist)
        labels_list_all = sorted(list(set(labels_list_all)))
        labels_map = {labelname:idx for idx, labelname in enumerate(labels_list_all)}
        length = len(labels_list_all)

        # generate itemlist
        res = []
        for index, row in df.iterrows():
            split_name = row[2]
            # if split_name != self.split and self.split != 'all':
            #     continue
            filename = row[0]
            imgpath = osp.join(self.img_dir, filename)
            label = [labels_map[i] for i in str_to_list(row[1])]
            label_np = np.zeros(length, dtype='float32')
            for idd in label:
                label_np[idd] = 1.0
            res.append((imgpath, label_np))

        return res

    def __len__(self) -> int:
        #if self.split == 'train':
        return len(self.train_labels)
        #elif self.split == 'val':
        #    return len(self.test_labels)
        #return self.num_samples

    def __getitem__(self, index: int):

        #if self.split == 'train':
        img_path, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img_path, target = self.test_data[index], self.test_labels[index]

        #imgpath, labels = self.itemlist[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def get_item_nuswide_numpy(self, index):
        # coco = self.coco
        # img_id = self.ids[index]
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        # target = coco.loadAnns(ann_ids)

        imgpath, targets = self.data[index]

        #output = np.zeros(self.num_classes)



        # for obj in target:
        #     output[self.cat2cat[obj['category_id']]] = 1
        # target = output

        # path = coco.loadImgs(img_id)[0]['file_name']

        return imgpath, targets


def str_to_list(text):
    """
    input: "['clouds', 'sky']" (str)
    output: ['clouds', 'sky'] (list)
    """
    # res = []
    res = [i.strip('[]\'\"\n ') for i in text.split(',')]
    return res