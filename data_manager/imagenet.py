import numpy as np
import os
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNet(Dataset):
    def __init__(self, root,
                 train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []
        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

        if train:
            train_dir = os.path.join(root, 'data', 'train')
            train_dset = datasets.ImageFolder(train_dir)
            for item in train_dset.imgs:
                self.data.append(item[0])
                self.targets.append(item[1])
        else:
            test_dir = os.path.join(root, 'data', 'val')
            test_dset = datasets.ImageFolder(test_dir)
            for item in test_dset.imgs:
                self.data.append(item[0])
                self.targets.append(item[1])
        self.data, self.targets = np.array(self.data), np.array(self.targets)

    def concatenate(self, datas, labels):
        con_data = datas[0].tolist()
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = con_data + datas[i].tolist()
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return np.array(con_data), con_label

    def get_test_data(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        paths, labels = self.concatenate(datas, labels)
        datas = []
        for path in paths:
            datas.append(pil_loader(path))
        self.test_data = datas if len(self.test_data) == 0 else self.test_data + datas
        self.test_targets = labels if len(self.test_targets) == 0 else np.concatenate((self.test_targets, labels), axis=0)

    def get_test_data_up2now(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        paths, labels = self.concatenate(datas, labels)
        datas = []
        for path in paths:
            datas.append(pil_loader(path))
        self.test_data = datas
        self.test_targets = labels

    def get_train_data(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        paths, labels = self.concatenate(datas, labels)
        datas = []
        for path in paths:
            datas.append(pil_loader(path))
        self.train_data = datas
        self.train_targets = labels

    def get_train_item(self, index):
        img, target = self.train_data[index], self.train_targets[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def get_test_item(self, index):
        img, target = self.test_data[index], self.test_targets[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def __getitem__(self, index):
        if len(self.train_data) != 0:
            return self.get_train_item(index)
        elif len(self.test_data) != 0:
            return self.get_test_item(index)

    def __len__(self):
        if len(self.train_data) != 0:
            return len(self.train_data)
        elif len(self.test_data) != 0:
            return len(self.test_data)

