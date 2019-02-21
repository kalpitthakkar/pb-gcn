import argparse
import os
import yaml

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Parser(object):
    def __init__(self):
        self.desc = "Spatio Temporal Graph Convolutional Resnet Training/Testing"
        self.parser = argparse.ArgumentParser(
                description=self.desc) 

    def create_parser(self):
        self.parser.add_argument(
            '--work-dir',
            default='./work_dir/temp',
            help='the work folder for storing results')
        self.parser.add_argument(
            '--config',
            default='./config/NTURGBD/nturgbd_xsub_train.yaml',
            help='path to the configuration file')

        # Trainer / Tester
        self.parser.add_argument(
            '--phase', default='train', help='must be train or test')
        self.parser.add_argument(
            '--save-score',
            type=str2bool,
            default=False,
            help='if True, the classification score will be stored')

        # print/save debugging information flags
        self.parser.add_argument(
            '--comment', type=str, default='', help='Comment to append to summary writer directory')
        self.parser.add_argument(
            '--seed', type=int, default=1, help='random seed for Pytorch')
        self.parser.add_argument(
            '--log-interval',
            type=int,
            default=100,
            help='the interval for printing messages (#iteration)')
        self.parser.add_argument(
            '--save-interval',
            type=int,
            default=10,
            help='the interval for storing models (#epochs)')
        self.parser.add_argument(
            '--eval-interval',
            type=int,
            default=5,
            help='the interval for evaluating models (#epochs)')
        self.parser.add_argument(
            '--print-log',
            type=str2bool,
            default=True,
            help='print logging or not')
        self.parser.add_argument(
            '--show-topk',
            type=int,
            default=[1, 5],
            nargs='+',
            help='which Top K accuracy will be shown')

        # loader
        self.parser.add_argument(
            '--loader', default='loader.NTULoader', help='data loader will be used')
        self.parser.add_argument(
            '--dataset', default='NTU', help='Which dataset is being used for running the program'
        )
        self.parser.add_argument(
            '--num-workers',
            type=int,
            default=32,
            help='the number of worker for data loader')
        self.parser.add_argument(
            '--train-loader-args',
            default=dict(),
            help='the arguments of training data loader')
        self.parser.add_argument(
            '--test-loader-args',
            default=dict(),
            help='the arguments of testing data loader')

        # model
        self.parser.add_argument('--model', default=None, help='the model will be used')
        self.parser.add_argument(
            '--model-args',
            type=dict,
            default=dict(),
            help='the arguments of model')
        self.parser.add_argument(
            '--weights',
            default=None,
            help='the weights for network initialization')
        self.parser.add_argument(
            '--ignore-weights',
            type=str,
            default=[],
            nargs='+',
            help='the name of weights which will be ignored in the initialization')

        # optim
        self.parser.add_argument(
            '--base-lr', type=float, default=0.01, help='initial learning rate')
        self.parser.add_argument(
            '--step',
            type=int,
            default=[20, 40, 60],
            nargs='+',
            help='the epoch where optimizer reduce the learning rate')
        self.parser.add_argument(
            '--device',
            type=int,
            default=0,
            nargs='+',
            help='the indices of GPUs for training or testing')
        self.parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        self.parser.add_argument(
            '--nesterov', type=str2bool, default=False, help='use nesterov or not')
        self.parser.add_argument(
            '--batch-size', type=int, default=64, help='training batch size')
        self.parser.add_argument(
            '--test-batch-size', type=int, default=64, help='test batch size')
        self.parser.add_argument(
            '--start-epoch',
            type=int,
            default=0,
            help='start training from which epoch')
        self.parser.add_argument(
            '--num-epoch',
            type=int,
            default=80,
            help='stop training in which epoch')
        self.parser.add_argument(
            '--weight-decay',
            type=float,
            default=0.0001,
            help='weight decay for optimizer')

    def dump_args(self, args, work_dir):
        arg_dict = vars(args)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        with open('{}/config.yaml'.format(work_dir), 'w') as f:
            yaml.dump(arg_dict, f)