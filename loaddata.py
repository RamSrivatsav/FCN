from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from skimage import io, transform
import os


class kittidata_split:
    def __init__(self, data_path, label_path, split_ratio):

        self.data = os.listdir(data_path)  # only store path rather than image, to save space
        self.data = list(map(lambda x: data_path + '/' + x, self.data))
        self.label = os.listdir(label_path)
        self.label = list(map(lambda x: label_path + '/' + x, self.label))

        self.total_size = len(self.data)
        self.train_size = int(self.total_size * split_ratio)

        self.train = self.data[:self.train_size]
        self.val = self.data[self.train_size:]
        self.train_label = self.label[:self.train_size]
        self.val_label = self.label[self.train_size:]

    def getdata(self, phase):
        if phase == 'train':
            return self.train, self.train_label
        else:
            return self.val, self.val_label


class kittidata(Dataset):
    def __init__(self, data, label, shrink_rate=0.6, flip_rate=0.5):
        self.class_name = ('sky', 'building', 'road', 'sidewalk', 'fence',
                           'vegetation', 'pole', 'car', 'sign', 'pedestrian',
                           'cyclist', 'ignore')
        self.class_color = ((128, 128, 128), (128, 0, 0), (128, 64, 128), (0, 0, 192), (64, 64, 128),
                            (128, 128, 0), (192, 192, 128), (64, 0, 128), (192, 128, 128), (64, 64, 0),
                            (0, 128, 192), (0, 0, 0))
        self.class_n = 12

        self.mean = [0.2902, 0.2976, 0.3042]
        self.std = [0.1271, 0.1330, 0.1431]

        self.data = data
        self.label = label

        self.flip_rate = flip_rate
        self.shrink_rate = shrink_rate
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = io.imread(self.data[idx])
        label = io.imread(self.label[idx])

        h, w, c = label.shape
        h = int(h // 32 * self.shrink_rate) * 32
        w = int(w // 32 * self.shrink_rate) * 32

        # use interpolation for quality
        img = transform.resize(img, (h, w), order=1, mode='constant', preserve_range=True).astype('uint8')
        # dont use interpolation to avoid illegel values
        label = transform.resize(label, (h, w), order=0, mode='constant', preserve_range=True).astype('uint8')

        if np.random.random() < self.flip_rate:
            img = np.fliplr(img).copy()  # cause error if remove '.copy()' (prevent memory sharing)
            label = np.fliplr(label).copy()

        img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])(img)

        num_label = torch.zeros(h, w).view(-1).long()
        # count = 0
        for i, v in enumerate(label.reshape(-1, c)):
            try:
                num_label[i] = self.class_color.index(tuple(v[:3]))  # some images are RGBA but some are RGB
            except:  # few pixel values not follow the defined labels above.
                # print(tuple(v[:3]))
                num_label[i] = 0  # it is not good yet
                # count += 1
        # print(count)
        num_label = num_label.view(h, w)

        target = torch.zeros(self.class_n, h, w)
        for c in range(self.class_n):
            target[c, num_label == c] = 1

        return img, target, num_label

    def visualize_list(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        label_list = list()
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)

        for label in labels:
            h, w = label.shape
            temp_label = np.zeros((h, w, 3), dtype='uint8')
            for i in range(h):  # how to write more elegantly
                for j in range(w):
                    temp_label[i, j] = self.class_color[int(label[i, j])]

            label_list.append(transforms.ToTensor()(temp_label))

        return label_list

    def visualize(self, label):
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)

        h, w = label.shape
        temp_label = np.zeros((h, w, 3), dtype='uint8')
        for i in range(h):  # how to write more elegantly
            for j in range(w):
                temp_label[i, j] = self.class_color[int(label[i, j])]

        return transforms.ToTensor()(temp_label)

    def denormalize(self, image):
        image = np.transpose(image, (1, 2, 0))
        image[:, :, 0] = image[:, :, 0] * self.std[0] + self.mean[0]
        image[:, :, 1] = image[:, :, 1] * self.std[1] + self.mean[1]
        image[:, :, 2] = image[:, :, 2] * self.std[2] + self.mean[2]
        return np.transpose(image, (2, 0, 1))