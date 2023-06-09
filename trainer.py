import torch
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime
from methods import RRFE


def set_logging_level(args, seed, log_path):
    if args['shuffle']:
        log_filename = log_path + '/seed={}_{}_{}_'.format(seed, args['backbone_type'], args['classifier_type'])
    else:
        log_filename = log_path + '/{}_{}_'.format(args['backbone_type'], args['classifier_type'])
    now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(
        format='%(asctime)s [%(filename)s]: %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
        filename=log_filename + now_time + '.log',
    )


def set_log_path(args):
    log_path = 'logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path + '/{}'.format(args['dataset'])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path + '/{}'.format(args['method_name'])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path + '/{}_{}'.format(args['init_cls'], args['task_num'])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return log_path


def set_save_path(args, seed):
    save_path = "model_saved_check"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + '/{}'.format(args['dataset'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + '/{}'.format(args['method_name'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args['shuffle']:
        save_path = save_path + '/seed={}_{}_{}'.format(seed, args['init_cls'], args['task_num'])
    else:
        save_path = save_path + '/{}_{}'.format(args['init_cls'], args['task_num'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path


def print_args(args):
    for k in args:
        args['logger'].info('{}: {}'.format(k, args[k]))


def train(args):
    for run, order_seed in enumerate(args['order_seed']):
        # set model path
        args.update({'save_path': set_save_path(args, order_seed)})
        # set log path
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        log_path = set_log_path(args)
        set_logging_level(args, order_seed, log_path)
        args.update({'logger': logger})

        # weather shuffle order
        class_order = [i for i in range(args['total_cls'])]
        if args['shuffle']:
            np.random.seed(order_seed)
            np.random.shuffle(class_order)
        label_maps = [class_order.index(i) for i in range(args['total_cls'])]
        args.update({'class_order': class_order, 'label_maps': label_maps})
        print_args(args)
        args['logger'].info("Launching seed {}/{}".format(run + 1, len(args['order_seed'])))

        if args['method_name'] == 'rrfe':
            trainer = RRFE(args)
        else:
            raise Exception('Can not find the method')

        t_start = datetime.now()
        accuracy = {}
        for phase in range(args['task_num'] + 1):
            args['logger'].info('Training task {} =======================>'.format(phase))
            accuracy = trainer.train(phase)
            t_end = datetime.now()
            args['logger'].info("Run time {}".format(t_end - t_start))
        args['logger'].info('================== Record ==================')
        args['logger'].info('stage accuracy: {}'.format(accuracy['avg_accs']))
        args['logger'].info('average accuracy: {:.2f}'.format(np.around(np.mean(accuracy['avg_accs']), 2)))
        max_acc = np.max(accuracy['stage_accs'][:-1], axis=0)[:-1]
        final_acc = accuracy['stage_accs'][:-1][-1][:-1]
        ft = np.mean(max_acc - final_acc)
        args['logger'].info(f'forgetting: {ft}')


