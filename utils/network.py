import torch
import torch.nn as nn
from models import resnet18_rrfe, resnet18_cbam
from models import LinearClassifier, CosineClassifier


def get_backbone(args, pretrained=False):
    if args['backbone_type'] == 'resnet18_rrfe':
        return resnet18_rrfe(pretrained, args=args)
    elif args['backbone_type'] == 'resnet18_cbam':
        return resnet18_cbam(pretrained, args=args)
    else:
        raise Exception('Unknown classifier type')


def get_classifier(args):
    in_features = 512
    out_features = args['init_cls']

    if args['classifier_type'] == 'linear':
        return LinearClassifier(in_features, out_features)
    elif args['classifier_type'] == 'cosine':
        return CosineClassifier(in_features, out_features)
    else:
        raise Exception('Unknown backbone type')


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.backbone = get_backbone(args)
        self.classifier = get_classifier(args)

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features['features'])
        return {'fmaps': features['fmaps'], 'features': features['features'], 'logits': output['logits']}

