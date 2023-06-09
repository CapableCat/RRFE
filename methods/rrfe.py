import torch
import numpy as np
import os
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
from methods.base import BaseLearner
from utils.network import Network
from utils.lookahead import Lookahead
from utils.model_para import filter_para
from utils.transform import rotation_transform


class RRFE(BaseLearner):
    """
    PyTorch's implementation of Representation Robustness and Feature Expansion
    """
    def __init__(self, args):
        super(RRFE, self).__init__(args)

    def train(self, phase):
        self.before_train(phase)

        model_path = self.save_path + "/{}.pt".format(self.num_cls)
        self.model.classifier.inc_cls(self.num_cls)
        if not self.args['resume'] and os.path.isfile(model_path):
            self.args.update({'backbone_type': 'resnet18_rrfe'})
            self.model = Network(args=self.args)
            para_dict = torch.load(model_path, map_location=self.device)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
            self.model.to(self.device)
            self.args['logger'].info('Load the saved model')
        else:
            if phase == 0:
                self.model.classifier.inc_cls(self.num_cls * 4)
            self.model.to(self.device)
            self.model.train()
            epochs = self.args['epochs'] if phase > 0 else self.args['init_epochs']
            weight_decay = self.args['weight_decay'] if phase > 0 else self.args['init_weight_decay']
            step_size = self.args['step_size'] if phase > 0 else self.args['init_step_size']
            if phase == 0 and self.args['sgd_fst'] is True:
                base_lr = 0.1
                step_size = 50
                epochs = 150
                custom_weight_decay = 1e-4
                custom_momentum = 0.9
                opt = torch.optim.SGD(self.model.parameters(), lr=base_lr, momentum=custom_momentum,
                                      weight_decay=custom_weight_decay)
            else:
                param = filter_para(self.model, self.args, phase)
                opt = torch.optim.Adam(param, weight_decay=weight_decay)
            opt = Lookahead(opt)
            scheduler = StepLR(opt, step_size=step_size, gamma=self.args['gamma'])
            for epoch in range(epochs):
                losses = 0.
                pbar = tqdm(self.train_loader, desc='Train T{}/{}, E{}/{}'.
                            format(phase + 1, self.task_num + 1, epoch + 1, epochs), ascii=True, ncols=120)
                for step, (index, images, labels) in enumerate(pbar):
                    images = images.to(self.device)
                    labels = torch.tensor(self.label_maps)[labels].to(self.device)
                    if phase == 0:
                        images, labels = rotation_transform(images, labels, images.shape[-1], 4)
                    opt.zero_grad()
                    loss = self.compute_loss(images, labels)
                    losses += loss.item()
                    loss.backward()
                    opt.step()
                    pbar.set_postfix(loss='{:.6f}'.format(losses / (step + 1)))
                if phase % 20 == 0:
                    self.test(phase, epoch, epochs)
                scheduler.step()

            if phase == 0:
                self.model.classifier.reshape(4)
                model = copy.deepcopy(self.model)
                para_dict = model.state_dict()
                para_dict_re = self.structure_reorganization(para_dict)
                self.args.update({'backbone_type': 'resnet18_rrfe'})
                self.model = Network(args=self.args)
                model_dict = self.model.state_dict()
                state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.model.load_state_dict(model_dict)
                self.model.to(self.device)
            torch.save(self.model.state_dict(), model_path)
        self.update_prototype(phase)
        self.after_train(phase)
        return self.accuracy

    def compute_loss(self, images, labels):
        output = self.model(images)
        loss_cls = nn.CrossEntropyLoss()(output['logits'] / self.args['temp'], labels)
        if self.old_model is None:
            return loss_cls
        else:
            output_old = self.old_model(images)
            loss_kd = torch.dist(output['features'], output_old['features'], 2)

            lam = 0.5
            proto_aug = []
            proto_mix = []
            proto_aug_label = []
            proto_mix_label_a = []
            proto_mix_label_b = []
            index = list(range(len(self.old_class)))
            for _ in range(self.args['batch_size']):
                np.random.shuffle(index)
                if index[0] >= self.args['init_cls']:
                    temp = self.prototype[index[0]]
                    proto_aug.append(temp)
                    proto_aug_label.append(index[0])
                else:
                    temp_a = self.prototype[index[0]]
                    num = list(filter(lambda x: x < self.args['init_cls'], index[1:]))[0]
                    temp_b = self.prototype[num]
                    temp = lam * temp_a + (1 - lam) * temp_b
                    proto_mix.append(temp)
                    proto_mix_label_a.append(index[0])
                    proto_mix_label_b.append(num)

            loss_protoAug = 0.
            if len(proto_aug):
                proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
                proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
                soft_feat_aug = self.model.classifier(proto_aug)['logits']
                loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.args['temp'], proto_aug_label)

            loss_protoMix = 0.
            if len(proto_mix):
                proto_mix = torch.from_numpy(np.float32(np.asarray(proto_mix))).float().to(self.device)
                proto_mix_label_a = torch.from_numpy(np.asarray(proto_mix_label_a)).to(self.device)
                proto_mix_label_b = torch.from_numpy(np.asarray(proto_mix_label_b)).to(self.device)
                soft_feat_mix = self.model.classifier(proto_mix)['logits']
                loss_protoMix = nn.CrossEntropyLoss()(soft_feat_mix / self.args['temp'], proto_mix_label_a) * lam + \
                                nn.CrossEntropyLoss()(soft_feat_mix / self.args['temp'], proto_mix_label_b) * (1 - lam)

            loss_proto = loss_protoAug + loss_protoMix
            return loss_cls + self.args['lambda_proto'] * loss_proto + self.args['lambda_proto'] * loss_kd

    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'bn1.weight' in k or 'bn2.weight' in k or 'downsample.1.weight' in k:
                if 'bn1.weight' in k:
                    k_conv3 = k.replace('bn1', 'conv1')
                elif 'bn2.weight' in k:
                    k_conv3 = k.replace('bn2', 'conv2')
                elif 'downsample.1.weight' in k:
                    k_conv3 = k.replace('1', '0')
                k_conv3_bias = k_conv3.replace('weight', 'bias')
                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re
