import os
import time
import numpy as np
import pickle

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import utils
from . import signals

class NTULoader(Dataset):
    """ Dataset loader class for the NTURGB+D dataset
    Constructor arguments:
        split_dir (type string) ->
            The directory where the train and test files for present split are located
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
                 transforms=None,
                 transform_args=dict(),
                 is_training=True,
                 signals=dict(),
                 window_size=-1):
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

        if is_training:
            self.data_path = os.path.join(split_dir, 'train_data.npy')
            self.label_path = os.path.join(split_dir, 'train_label.pkl')
        else:
            self.data_path = os.path.join(split_dir, 'val_data.npy')
            self.label_path = os.path.join(split_dir, 'val_label.pkl')

        """ Load the labels from the pickle file
            Each record -> (sample_name, label)
        """
        try:
            with open(self.label_path, 'r') as f:
                self.sample_name, self.labels = pickle.load(f)
        except Exception as e:
            print("The pickled file seems to be created with Python2.x: ", e)
            print("Opening the file with 'rb' flags")
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
        label = self.labels[index]

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
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            sample = np.concatenate([sample, disps, rel_coords], axis=0)
        elif self.temporal_signal and self.spatial_signal:
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            sample = np.concatenate([disps, rel_coords], axis=0)
        elif self.temporal_signal:
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            sample = disps
        elif self.spatial_signal:
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            sample = rel_coords

        return sample, label

    def __len__(self):
        return self.samples.shape[0]

    def __iter__(self):
        return self
