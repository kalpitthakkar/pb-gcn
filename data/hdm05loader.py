import os
import time
import numpy as np
import pickle

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import utils
import signals

def split_augment_data(data):
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    data = data[:, start:end, :, :]
    C, T, V, M = data.shape

    num_frames = data.shape[1]
    num_chunks = num_frames / 8
    last_chunk_size = (num_frames % 8 if num_frames % 8 > 0 else 8)

    sequences = np.zeros(8, C, 120, V, M)
    for n in range(8):
        seq = []
        for i in range(num_chunks):
            if i == num_chunks-1:
                frame_idx = np.random.randint(0, last_chunk_size, 1)
            else:
                frame_idx = np.random.randint(0, 8, 1)
            seq.append(data[:, i*8+frame_idx, :, :])

        seq = np.concatenate(seq, axis=1)
        sequences[n, :, :num_chunks, :, :] = seq

    return sequences

class HDM05Loader(Dataset):
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

        """ Load the labels from the pickle file
            Each record -> (sample_name, label)
        """
        try:
            with open(self.label_path, 'r') as f:
                self.sample_name, self.labels = pickle.load(f)
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

        if split_name == '10xsam':
            # Perform the random sample split. Half from each class for training and 
            # rest half from each class for testing.
            # Set the random state when creating test loader to be consistent with train.
            if random_state is not None:
                np.random.set_state(random_state)
            np.random.shuffle(self.samples)
            if random_state is not None:
                np.random.set_state(random_state)
            np.random.shuffle(self.labels)

            train_idx = []
            test_idx = []
            for i in range(130):
                class_idx = np.where(np.isin(self.labels, i))
                num_idx = len(class_idx[0])
                add = num_idx % 2
                train_idx.extend(class_idx[0][:num_idx/2+add])
                test_idx.extend(class_idx[0][num_idx/2+add:])

            if is_training:
                train_idx = np.asarray(train_idx)
                self.samples = self.samples[train_idx, :, :, :, :]
                self.labels = np.asarray(self.labels)[train_idx]
            else:
                test_idx = np.asarray(test_idx)
                self.samples = self.samples[test_idx, :, :, :, :]
                self.labels = np.asarray(self.labels)[test_idx]
        else:
            train_subs = ['bd', 'mm']
            test_subs = ['bk', 'dg', 'tr']
            train_idx = np.where(np.isin(self.subjects, train_subs))
            test_idx = np.where(np.isin(self.subjects, test_subs))

            if is_training:
                self.samples = self.samples[train_idx[0], :, :, :]
                self.labels = np.asarray(self.labels)[train_idx[0]]
            else:
                self.samples = self.samples[test_idx[0], :, :, :]
                self.labels = np.asarray(self.labels)[test_idx[0]]

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
            #orient_disps = getattr(signals, 'orientedDisplacements')(sample=sample)
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample)
            #rel_angles = getattr(signals, 'relativeAngularCoordinates')(sample=sample)
            sample = np.concatenate([sample, disps, rel_coords], axis=0)
        elif self.temporal_signal and self.spatial_signal:
            #orient_disps = getattr(signals, 'orientedDisplacements')(sample=sample)
            disps = getattr(signals, 'displacementVectors')(sample=sample)
            rel_coords = getattr(signals, 'relativeCoordinates')(sample=sample, references=(19, 26, 2, 7))
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

        #sample = split_augment_data(sample)
        #label = np.asarray([label] * 8)

        return sample, label

    def __len__(self):
        return self.samples.shape[0]

    def __iter__(self):
        return self
