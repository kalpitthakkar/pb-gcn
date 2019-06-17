from __future__ import print_function
import os, glob
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils import Parser, Timer
from utils.nturgbd import gendata as ntu_gendata
from utils.hdm05 import gendata as hdm_gendata
import data
import models

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Runner(object):
    """
    Class having all the required methods to start training and testing
    """

    def __init__(self, args):
        self.args = args
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.best_valid = []

    def load_data(self):
        if 'NTU' in self.args.dataset:
            if self.args.phase == 'train':
                self.train_dataset = getattr(data, self.args.loader)(**self.args.train_loader_args)
                self.train_loader = DataLoader(self.train_dataset,
                    batch_size=self.args.batch_size, shuffle=True,
                    num_workers=self.args.num_workers, pin_memory=True)
            self.test_dataset = getattr(data, self.args.loader)(**self.args.test_loader_args)
            self.test_loader = DataLoader(self.test_dataset,
                batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True)
        elif 'HDM' in self.args.dataset:
            st = np.random.get_state()
            extra_arg = dict(random_state=st)
            self.args.train_loader_args.update(extra_arg)
            self.train_dataset = getattr(data, self.args.loader)(**self.args.train_loader_args)
            self.train_loader = DataLoader(self.train_dataset,
                batch_size=self.args.batch_size, shuffle=True,
                num_workers=self.args.num_workers, pin_memory=True)
            extra_arg = dict(random_state=st)
            self.args.test_loader_args.update(extra_arg)
            self.test_dataset = getattr(data, self.args.loader)(**self.args.test_loader_args)
            self.test_loader = DataLoader(self.test_dataset,
                batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True)

    def load_model(self):
        output_device = self.args.device[
            0] if type(self.args.device) is list else self.args.device
        self.output_device = output_device
        Model = getattr(models, self.args.model)
        self.model = Model(**self.args.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.args.weights:
            self.print_log('Load weights from {}.'.format(self.args.weights))
            self.weights = torch.load(self.args.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in self.weights.items()])

            for w in self.args.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if isinstance(self.args.device, list):
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()

    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            lr = self.args.base_lr * (
                0.1**np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args.print_log:
            with open('{}/log.txt'.format(self.args.work_dir), 'a') as f:
                print(str, file=f)

    def top_k(self, score, label, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def train(self, epoch, save_model=False):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        lr = self.adjust_learning_rate(epoch)
        loss_value = []

        timetracker = Timer(print_log=self.args.print_log, work_dir=self.args.work_dir)
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        for batch_idx, (data, label) in enumerate(self.train_loader):

            # get data
            data = Variable(
                data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(
                label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += timetracker.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())
            score_frag = output.data.cpu().numpy()
            label_frag = label.data.cpu().numpy()
            timer['model'] += timetracker.split_time()

            hit1 = self.top_k(score_frag, label_frag, 1)
            hit5 = self.top_k(score_frag, label_frag, 5)
            loss_val = loss.item()

            losses.update(loss_val, data[0].size(0))
            top1.update(hit1 * 100., data[0].size(0))
            top5.update(hit5 * 100., data[0].size(0))

            # statistics
            if batch_idx % self.args.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Top1: {:.2f} ({:.2f})  Top5: {:.2f} ({:.2f}) '
                    ' Loss: {:.4f} ({:.4f})  lr:{:.6f}'.format(
                        batch_idx, len(self.train_loader), top1.val, top1.avg,
                        top5.val, top5.avg, losses.val, losses.avg, lr))
                step = epoch * len(self.train_loader) + batch_idx
                self.summary_writer.add_scalar('Train/AvgLoss', losses.avg, step)
                self.summary_writer.add_scalar('Train/AvgTop1', top1.avg, step)
                self.summary_writer.add_scalar('Train/AvgTop5', top5.avg, step)
                self.summary_writer.add_scalar('Train/LearningRate', lr, step)
            timer['statistics'] += timetracker.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            model_path = '{}/epoch{}_model.pt'.format(self.args.work_dir,
                                                      epoch + 1)
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, model_path)

    def eval(self, epoch, save_score=False):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        loss_value = []
        score_frag = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.test_loader):
                data = Variable(
                    data.float().cuda(self.output_device),
                    requires_grad=False)
                label = Variable(
                    label.long().cuda(self.output_device),
                    requires_grad=False)
                output = self.model(data)
                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.item())
            score = np.concatenate(score_frag)
            score_dict = dict(
                zip(self.test_dataset.sample_name, score))
            self.print_log('\tMean test loss of {} batches: {}.'.format(
                len(self.test_loader), np.mean(loss_value)))
            self.summary_writer.add_scalar('Test/AvgLoss', np.mean(loss_value), epoch)
            for k in self.args.show_topk:
                hit_val = self.top_k(score, self.test_dataset.labels, k)
                self.summary_writer.add_scalar('Test/AvgTop'+str(k), hit_val, epoch)
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * hit_val))
                if k == 1:
                    self.best_valid.append((np.mean(loss_value), 100*hit_val))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.args.work_dir, epoch + 1, 'test'), 'w') as f:
                    pickle.dump(score_dict, f)

    def run(self):
        self.logdir = os.path.join(self.args.work_dir, 'runs')
        if not self.args.comment == '':
            self.summary_writer = SummaryWriter(logdir=self.logdir, comment='_'+self.args.comment)
        else:
            self.summary_writer = SummaryWriter(logdir=self.logdir)
        if self.args.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.args))))
            for epoch in range(self.args.start_epoch, self.args.num_epoch):
                save_model = ((epoch + 1) % self.args.save_interval == 0) or (
                    epoch + 1 == self.args.num_epoch)
                eval_model = ((epoch + 1) % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch)

                self.train(epoch, save_model=save_model)

                if eval_model:
                    self.eval(
                        epoch,
                        save_score=self.args.save_score)
                else:
                    pass

        elif self.args.phase == 'test':
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.args.print_log = False
            self.print_log('Model:   {}.'.format(self.args.model))
            self.print_log('Weights: {}.'.format(self.args.weights))
            self.eval(
                epoch=0, save_score=self.args.save_score)
            self.print_log('Done.\n')

if __name__ == '__main__':
    p = Parser()
    p.create_parser()

    pargs = p.parser.parse_args()
    if pargs.config is not None:
        with open(pargs.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(pargs).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        p.parser.set_defaults(**default_arg)

    args = p.parser.parse_args()
    p.dump_args(args, args.work_dir)

    if 'HDM' in args.dataset:
        # Prepare the data if not already present
        # No splits => prepared at runtime
        data_check = glob.glob(os.path.join(
            args.train_loader_args['split_dir'],
            'full_*')
        )
        print(data_check)
        if not (len(data_check) == 2):
            if not os.path.exists(args.train_loader_args['split_dir']):
                os.makedirs(args.train_loader_args['split_dir'])
            hdm_gendata(
                args.data_path,
                args.train_loader_args['split_dir']
            )
        # Run the 10-fold cross-validation on HDM05
        cv_acc = 0.
        for i in range(10):
            launcher = Runner(args)
            launcher.run()
            best_valid_loss = sorted(self.best_valid, key=lambda x: x[0])
            best_valid_acc = sorted(self.best_valid, key=lambda x: -x[1])
            print("Lowest loss value (accuracy): {} ({})".format(best_valid_loss[0][0], best_valid_loss[0][1]))
            print("Highest accuracy value (loss): {} ({})".format(best_valid_acc[0][1], best_valid_acc[0][0]))
            cv_acc += best_valid_acc[0][1]
        print("Final CV accuracy: {}".format(cv_acc / 10.))

    elif 'NTU' in args.dataset:
        if args.phase == 'train':
            # Prepare training data if it is not already present
            train_data_check = glob.glob(os.path.join(
                args.train_loader_args['split_dir'],
                'train_*')
            )
            print(train_data_check)
            if not (len(train_data_check) == 2):
                topdir, benchmark = os.path.split(args.train_loader_args['split_dir'])
                part = 'train'
                if not os.path.exists(args.train_loader_args['split_dir']):
                    os.makedirs(args.train_loader_args['split_dir'])
                ntu_gendata(
                    args.data_path,
                    args.train_loader_args['split_dir'],
                    args.missing_txt,
                    benchmark=benchmark,
                    part=part
                )

        # Prepare testing data if it is not already present
        test_data_check = glob.glob(os.path.join(
            args.test_loader_args['split_dir'],
            'val_*')
        )
        print(test_data_check)
        if not (len(test_data_check) == 2):
            topdir, benchmark = os.path.split(args.test_loader_args['split_dir'])
            part = 'val'
            if not os.path.exists(args.test_loader_args['split_dir']):
                os.makedirs(args.test_loader_args['split_dir'])
            ntu_gendata(
                args.data_path,
                args.test_loader_args['split_dir'],
                args.missing_txt,
                benchmark=benchmark,
                part=part
            )

        # Launch the training process
        launcher = Runner(args)
        launcher.run()
        best_valid_loss = sorted(self.best_valid, key=lambda x: x[0])
        best_valid_acc = sorted(self.best_valid, key=lambda x: -x[1])
        print("Lowest loss value (accuracy): {} ({})".format(best_valid_loss[0][0], best_valid_loss[0][1]))
        print("Highest accuracy value (loss): {} ({})".format(best_valid_acc[0][1], best_valid_acc[0][0]))
