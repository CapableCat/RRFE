from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image


class Cifar100(CIFAR100):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(Cifar100, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                       download=download)
        self.transform = transform
        self.target_transform = target_transform
        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def get_test_data(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.test_data = datas if len(self.test_data) == 0 else np.concatenate((self.test_data, datas), axis=0)
        self.test_targets = labels if len(self.test_targets) == 0 else np.concatenate((self.test_targets, labels), axis=0)

    def get_test_data_up2now(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.test_data = datas
        self.test_targets = labels

    def get_train_data(self, classes):
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.train_data, self.train_targets = self.concatenate(datas, labels)

    def get_train_item(self, index):
        img, target = Image.fromarray(self.train_data[index]), self.train_targets[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def get_test_item(self, index):
        img, target = Image.fromarray(self.test_data[index]), self.test_targets[index]
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

