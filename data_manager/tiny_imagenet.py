import numpy as np
import os
from torchvision import datasets
from PIL import Image
import sys
from torch.utils.data import Dataset


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        train_dir = os.path.join(root, 'train')
        test_dir = os.path.join(root, 'val')

        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []
        self.train_data = []
        self.train_targets = []
        self.test_data = []
        self.test_targets = []

        if train:
            train_dset = datasets.ImageFolder(train_dir)
            for item in train_dset.imgs:
                self.data.append(np.array(pil_loader(item[0])))
                self.targets.append(item[1])
        else:
            test_images = []
            _, class_to_idx = find_classes(train_dir)
            imgs_path = os.path.join(test_dir, 'images')
            imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())
            cls_map = {line_data[0]: line_data[1] for line_data in data_info}
            for imgname in sorted(os.listdir(imgs_path)):
                if cls_map[imgname] in sorted(class_to_idx.keys()):
                    path = os.path.join(imgs_path, imgname)
                    test_images.append(path)
                    self.targets.append(class_to_idx[cls_map[imgname]])
            for i in range(len(test_images)):
                self.data.append(np.array(pil_loader(test_images[i])))
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

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
        self.test_targets = labels if len(self.test_targets) == 0 else np.concatenate((self.test_targets, labels),
                                                                                      axis=0)

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

