def filter_para(model, args, phase):
    if phase == 0:
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args['init_lr']}]
    else:
        return [{'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': args['lr']},
                {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': args['gamma']*args['lr']}]
