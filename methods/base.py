import numpy as np
from utils.network import Network
import torch
from data_manager import ImageNet, TinyImageNet, Cifar100
from utils.transform import get_transform
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self.class_order = args['class_order']
        self.label_maps = args['label_maps']
        self.total_cls = args['total_cls']
        self.num_cls = args['init_cls']
        self.task_num = args['task_num']
        self.task_size = (self.total_cls - self.num_cls) // self.task_num if self.task_num > 0 else 0
        assert self.task_size * self.task_num == self.total_cls - self.num_cls, \
            'tasks cannot be divided by remaining classes! ' \
            'total_cls={}, init_cls={}, task_size={}'.format(self.total_cls, self.num_cls, self.task_size)

        self.save_path = args['save_path']
        self.dataset = args['dataset']
        # Define memory and prototype
        self.prototype = []
        self.prototype_labels = []
        # Define model
        self.model = Network(args)
        self.old_model = None
        self.device = torch.device('cuda:' + args['gpu'] if torch.cuda.is_available() else "cpu")
        # Prepare data
        self.train_transform = get_transform(args['dataset'], phase='train')
        self.test_transform = get_transform(args['dataset'], phase='test')
        if args['dataset'] == 'cifar100':
            self.train_dataset = Cifar100(root=args['root'], transform=self.train_transform)
            self.test_dataset = Cifar100(root=args['root'], train=False, transform=self.test_transform)
            self.eval_dataset = Cifar100(root=args['root'], train=False, transform=self.test_transform)
        elif args['dataset'] == 'tinyimagenet':
            self.train_dataset = TinyImageNet(root=args['root'], transform=self.train_transform)
            self.test_dataset = TinyImageNet(root=args['root'], train=False, transform=self.test_transform)
            self.eval_dataset = TinyImageNet(root=args['root'], train=False, transform=self.test_transform)
        elif args['dataset'] == 'imagenet' or args['dataset'] == 'imagenet_subset':
            self.train_dataset = ImageNet(root=args['root'], transform=self.train_transform)
            self.test_dataset = ImageNet(root=args['root'], train=False, transform=self.test_transform)
            self.eval_dataset = ImageNet(root=args['root'], train=False, transform=self.test_transform)
        else:
            raise Exception('Can not find the dataset')

        self.old_class = []
        self.inc_class = []
        self.classes = []
        self.train_loader = None
        self.test_loader = None

        self.accuracy = {}

    def before_train(self, phase):
        # ---------------
        # 1. Prepare Task
        # ---------------
        self.old_class = [] if phase == 0 else self.class_order[:self.args['init_cls'] + (phase - 1) * self.task_size]
        self.inc_class = self.class_order[:self.args['init_cls']] if phase == 0 else \
            self.class_order[self.args['init_cls'] + (phase - 1) * self.task_size:
                             self.args['init_cls'] + phase * self.task_size]
        self.classes.append(self.inc_class)
        self.test_dataset.get_test_data(self.inc_class)
        print('The size of train set is   {}'.format(len(self.train_dataset.train_data)))
        print('The size of test set is    {}'.format(len(self.test_dataset.test_data)))
        self.test_loader = DataLoader(dataset=self.test_dataset, shuffle=False,
                                      batch_size=self.args['batch_size'], num_workers=self.args['num_workers'])

    def train(self, phase):
        pass

    def after_train(self, phase):
        self.args['logger'].info('Validation:')
        self.args['logger'].info('[Task: %d, numImages: %5d]' % (phase, len(self.test_dataset.test_data)))
        self.compute_accuracy()
        self.args['logger'].info('average accuracy: {}'.format(self.accuracy['avg_accs'][-1]))
        self.args['logger'].info('stage accuracy: {}'.format(self.accuracy['stage_accs'][-1]))
        self.num_cls += self.task_size
        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()
        return self.accuracy

    def test(self, phase, epoch, epochs):
        self.model.eval()
        pbar = tqdm(self.test_loader, desc='Test  T{}/{}, E{}/{}'.
                    format(phase + 1, self.task_num + 1, epoch + 1, epochs), ascii=True, ncols=120)
        correct_old, total_old = 0.0, 0.0
        correct_new, total_new = 0.0, 0.0
        for step, (indexs, images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = torch.tensor(self.label_maps)[labels].to(self.device)
            with torch.no_grad():
                output = self.model(images)
            ssl = 4 if phase == 0 else 1
            predicts = torch.max(output['logits'][:, ::ssl], dim=1)[1]
            pos_old = labels < len(self.old_class)
            correct_old += (predicts[pos_old].cpu() == labels[pos_old].cpu()).sum()
            total_old += pos_old.cpu().sum()

            pos_new = labels >= len(self.old_class)
            correct_new += (predicts[pos_new].cpu() == labels[pos_new].cpu()).sum()
            total_new += pos_new.cpu().sum()

            correct_all = correct_old + correct_new
            total_all = total_old + total_new

            pbar.set_postfix(old_acc='{:.2f}%'.format(correct_old.item() / total_old * 100),
                             new_acc='{:.2f}%'.format(correct_new.item() / total_new * 100),
                             all_acc='{:.2f}%'.format(correct_all.item() / total_all * 100))
        self.model.train()

    def compute_accuracy(self):
        self.model.eval()

        total_correct, total_nums = 0, 0
        stage_accuracy = []
        for cur_class in self.classes:
            self.eval_dataset.get_test_data_up2now(cur_class)
            test_loader = DataLoader(dataset=self.eval_dataset, shuffle=False, batch_size=self.args['batch_size'])
            correct, nums = 0.0, 0.0
            for step, (index, images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = torch.tensor(self.label_maps)[labels].to(self.device)
                with torch.no_grad():
                    output = self.model(images)
                nums += len(labels)
                # Compute score for CNN
                predicts = torch.max(output['logits'], dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()

            accuracy_cur = np.around(correct.item() / nums * 100, decimals=2)
            stage_accuracy.append(accuracy_cur)
            total_correct += correct
            total_nums += nums

        avg_acc = np.around(total_correct.item() / total_nums * 100, decimals=2)
        stage_accuracy.extend((self.task_num + 1 - len(self.classes)) * [0])
        accuracy = {'avg_acc': avg_acc, 'stage_acc': stage_accuracy}

        # record accuracy
        if self.num_cls == self.args['init_cls']:
            avg_accs, stage_accs = [], []
        else:
            avg_accs, stage_accs = self.accuracy['avg_accs'], self.accuracy['stage_accs']
        avg_accs.append(accuracy['avg_acc'])
        stage_accs.append(accuracy['stage_acc'])
        self.accuracy.update({'avg_accs': avg_accs, 'stage_accs': stage_accs})

    def extract_features(self, model, data_loader):
        model.eval()

        features = []
        labels = []
        for i, (index, images, targets) in enumerate(data_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            with torch.no_grad():
                feature = model.backbone(images.to(self.device))['features']

            if len(features) == 0:
                features = feature.cpu().numpy()
                labels = targets.cpu().numpy()
            else:
                features = np.vstack((features, feature.cpu().numpy()))
                labels = np.hstack((labels, targets.cpu().numpy()))

        return features, labels

    def update_prototype(self, phase):
        features, labels = self.extract_features(self.model, self.train_loader)
        prototype = []
        for item in self.inc_class:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
        if phase == 0:
            self.prototype = prototype
            self.prototype_labels = self.inc_class
        else:
            self.prototype = self.prototype + prototype
            self.prototype_labels = self.prototype_labels + self.inc_class


