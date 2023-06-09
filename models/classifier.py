import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        out = self.fc(x)
        return {"logits": out}

    def inc_cls(self, num_cls):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        self.fc = nn.Linear(in_features, num_cls, bias=True)
        self.fc.weight.data[:out_features] = weight[:out_features]
        self.fc.bias.data[:out_features] = bias[:out_features]

    def reshape(self, ssl):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        num_cls = out_features // ssl
        assert num_cls * ssl == out_features, "can not divide"
        self.fc = nn.Linear(in_features, num_cls, bias=True)
        self.fc.weight.data = weight[::ssl]
        self.fc.bias.data = bias[::ssl]


class CosineClassifier(nn.Module):
    def __init__(self, outplanes, numclass):
        super(CosineClassifier, self).__init__()
        self.numclass = numclass
        self.fc = Cosine(outplanes, numclass)

    def forward(self, x):
        return self.fc(x)

    def inc_class(self, numclass):
        weight = self.fc.weight.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = Cosine(in_feature, numclass)
        self.fc.weight.data[:out_feature] = weight[:out_feature]

    def reshape(self, numclass, ssl):
        weight = self.fc.weight.data
        in_feature = self.fc.in_features

        self.fc = Cosine(in_feature, numclass)
        self.fc.weight.data = weight[::ssl]


class Cosine(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(Cosine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, x):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

