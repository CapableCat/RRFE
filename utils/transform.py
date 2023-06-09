import torch
from torchvision import transforms


def get_transform(data_name='cifar100', phase='test'):
    transform_list = []
    if phase == 'train':
        if data_name == 'cifar100':
            transform_list.extend([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.AugMix(severity=5, chain_depth=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        if data_name == 'tinyimagenet':
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.AugMix(severity=5, chain_depth=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if data_name == 'imagenet' or data_name == 'imagenet_subset':
            transform_list.extend([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AugMix(severity=5, chain_depth=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    if phase == 'test':
        if data_name == 'cifar100':
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        if data_name == 'tinyimagenet':
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if data_name == 'imagenet' or data_name == 'imagenet_subset':
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    return transforms.Compose(transform_list)


def rotation_transform(images, labels, hw, ssl):
    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(ssl)], 1)
    images = images.view(-1, 3, hw, hw)
    labels = torch.stack([labels * ssl + k for k in range(ssl)], 1).view(-1)
    return images, labels