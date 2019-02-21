import os
import time
import numpy as np
import pickle

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import utils
import signals

# Head: 3, 4, 5, 9 / 1, 2, 4, 6
head = [[(2, 3), (2, 4), (2, 8)], [(0, 1), (1, 3), (1, 5)]]
# LHand: 5, 6, 7, 8 / 4, 5, 12
lefthand = [[(4, 5), (5, 6), (6, 7)], [(3, 4), (4, 11)]]
# RHand: 9, 10, 11, 12 / 6, 7, 13
righthand = [[(8, 9), (9, 10), (10, 11)], [(5, 6), (6, 12)]]
hands = [lefthand[i] + righthand[i] for i in range(len(num_node))]
# Torso: 1, 2, 3, 5, 9, 13, 17 / 2, 3, 4, 6, 8, 10
torso = [[(0, 1), (1, 2), (2, 4), (2, 8), (0, 12), (0, 16)], [(1, 2), (1, 3), (1, 5), (2, 7), (2, 9)]]
# Lleg: 1, 13, 14, 15, 16 / 3, 8, 9, 14
leftleg = [[(0, 12), (12, 13), (13, 14), (14, 15)], [(2, 7), (7, 8), (8, 13)]]
# Rleg: 1, 17, 18, 19, 20 / 3, 10, 11, 15
rightleg = [[(0, 16), (16, 17), (17, 18), (18, 19)], [(2, 9), (9, 10), (10, 14)]]
legs = [leftleg[i] + rightleg[i] for i in range(len(num_node))]

partlist = [[head[i], hands[i], torso[i], legs[i]] for i in range(2)]

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_part_masks(num_node, parts):
    stack = []
    for p in parts:
        stack.append(edge2mat(p, num_node))
    masks = np.int64(np.stack(stack))
    return masks

class LSCLoader(Dataset):
    """ Dataset loader class for the NTURGB+D dataset
    Constructor arguments:
        split_dir (type string) ->
            The directory where the full dataset file is located
        transforms (type list) ->
            The name of the transforms to be applied for augmentation
        transform_args (type dict) ->
            The extra arguments that are necessary for the transformations to be applied
        is_training (type bool) ->
            Whether the current phase is training or testing
        signals (type dict) ->
            Which extra signals to use other than 3D joint locations
        window_size (type int) ->
            The number of frames in each sample to be loaded
    """
    def __init__(self,
                 split_dir,
                 split_name='xsub',
                 transforms=None,
                 transform_args=dict(),
                 is_training=True,
                 signals=dict(),
                 window_size=-1,
                 random_state=None):
        self.transforms = transforms
        self.split_dir = split_dir
        self.is_training = is_training
        self.num_frames = window_size
        self.transform_args = transform_args

        self.temporal_signal = False
        self.spatial_signal = False
        self.all_signal = False
        if 'temporal_signal' in signals.keys():
            self.temporal_signal = signals['temporal_signal']
        if 'spatial_signal' in signals.keys():
            self.spatial_signal = signals['spatial_signal']
        if 'all_signal' in signals.keys():
            self.all_signal = signals['all_signal']

        self.data_path = os.path.join(split_dir, 'full_data.npy')
        self.label_path = os.path.join(split_dir, 'full_label.pkl')
        self.adj_masks = []
        for ps in partlist:
            self.adj_masks.append(get_part_masks(20, ps))

        """ Load the labels from the pickle file
            Each record -> (sample_name, label)
        """
        try:
            with open(self.label_path, 'r') as f:
                self.sample_name, self.labels, self.subjects, self.joints = pickle.load(f)
        except Exception as e:
            print("The pickled file seems to be created with Python2.x: ", e)
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.labels = pickle.load(f, encoding='latin1')

        """ Load the data from the npy file. Shape of data -> (N, C, T, V, M)
            N : Number of videos
            C : Number of coordinates
            T : Number of frames
            V : Number of joints
            M : Number of actors
        """
        try:
            self.samples = np.load(self.data_path)
        except Exception as e:
            print("Error in loading the .npy file: ", e)

        # Perform the random sample split. Half for training and half for testing.
        # Set the random state when creating test loader to be consistent with train.
        if split_name == 'xsam':
            if random_state is not None:
                np.random.set_state(random_state)
            np.random.shuffle(self.samples)
            if random_state is not None:
                np.random.set_state(random_state)
            np.random.shuffle(self.labels)
            if random_state is not None:
                np.random.set_state(random_state)
            np.random.shuffle(self.joints)
        
            train_idx = []
            test_idx = []
            for i in range(1, 89):
                class_idx = np.where(np.isin(self.labels, i))
                num_idx = len(class_idx[0])
                add = num_idx % 2
                train_idx.extend(class_idx[0][:num_idx/2+add])
                test_idx.extend(class_idx[0][num_idx/2+add:])

            if is_training:
                train_idx = np.asarray(train_idx)
                self.samples = self.samples[train_idx, :, :, :, :]
                self.labels = np.asarray(self.labels)[train_idx]
                self.joints = np.asarray(self.joints)[train_idx]
            else:
                test_idx = np.asarray(test_idx)
                self.samples = self.samples[test_idx, :, :, :, :]
                self.labels = np.asarray(self.labels)[test_idx]
                self.joints = np.asarray(self.joints)[test_idx]
        else:
            num_subs = len(np.unique(self.subjects))
            if not (num_subs % 2 == 0):
                num_tr_subs = (num_subs/2)+1
                num_te_subs = (num_subs/2)
            else:
                num_tr_subs = num_te_subs = (num_subs/2)

            if random_state is not None:
                np.random.set_state(random_state)
            train_subs = np.random.choose(np.unique(self.subjects), num_tr_subs, replace=False)
            test_subs = np.asarray([x for x in np.unique(self.subjects) if x not in train_subs])

            train_idx = np.where(np.isin(self.subjects, train_subs))
            test_idx = np.where(np.isin(self.subjects, test_subs))

            if is_training:
                self.samples = self.samples[train_idx[0], :, :, :]
                self.labels = np.asarray(self.labels)[train_idx[0]]
                self.joints = np.asarray(self.joints)[train_idx[0]]
            else:
                self.samples = self.samples[test_idx[0], :, :, :]
                self.labels = np.asarray(self.labels)[test_idx[0]]
                self.joints = np.asarray(self.joints)[test_idx[0]]

    def get_mean_map(self):
        data = self.samples
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
                axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]-1
        njoints = self.joints[index]
        if njoints == 20:
            mask = self.adj_masks[0]
        else:
            mask = self.adj_masks[1]

        basic_args = dict(sample=sample, window_size=self.num_frames)
        transform_args = {}
        transform_args.update(basic_args)
        transform_args.update(self.transform_args)

        if self.transforms is not None:
            # The ordering of the transforms given in arguments is important
            for transform in self.transforms:
                trans_func = getattr(utils, transform)
                sample = trans_func(**transform_args)
                transform_args['sample'] = sample

        if self.all_signal:
            #orient_disps = getattr(signals, 'orientedDisplacements')(sample=sample)
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            #rel_angles = getattr(signals, 'relativeAngularCoordinates')(sample=sample)
            sample = np.concatenate([sample, disps, rel_coords], axis=0)
        elif self.temporal_signal and self.spatial_signal:
            #orient_disps = getattr(signals, 'orientedDisplacements')(sample=sample)
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            if njoints == 15:
                references = (3, 5, 7, 9)
            else:
                references = (4, 8, 12, 16)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample, references=references)
            #rel_angles = getattr(signals, 'relativeAngularCoordinates')(sample=sample)
            sample = np.concatenate([disps, rel_coords], axis=0)
        elif self.temporal_signal:
            orient_disps = getattr(signals, 'orientedDisplacements')(sample=sample)
            #disps = getattr(signals, 'displacementVectors')(sample=sample)
            sample = orient_disps
            #sample = np.concatenate([disps, orient_disps], axis=0)
        elif self.spatial_signal:
            #rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            rel_angles = getattr(signals, 'relativeAngularCoordinates')(sample=sample)
            sample = rel_angles
            #sample = np.concatenate([rel_coords, rel_angles], axis=0)

        return sample, label, mask

    def __len__(self):
        return self.samples.shape[0]

    def __iter__(self):
        return self
