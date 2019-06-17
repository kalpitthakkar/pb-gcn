import os
import glob
import pickle
from itertools import groupby

import argparse
from numpy.lib.format import open_memmap
from tqdm import tqdm

from .read_skeleton import read_xyz

max_body = 1
num_joint = 31
max_frame = 901

def gendata(data_path,
            out_path):
    sample_name = []
    sample_label = []
    filenames = sorted(glob.glob(os.path.join(data_path, '*/*.mat')))
    grouped_dict = {g[0]: list(g[1]) for g in groupby(filenames, key=lambda x: x.split('/')[-2])}
    for idx, (k, v) in enumerate(grouped_dict.items()):
        for filename in v:
            action_class = idx

            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, 'full'), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, 'full'),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i in tqdm(range(len(sample_name))):
        s = sample_name[i]
        data = read_xyz(s, max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HDM05 Data Converter.')
    parser.add_argument(
        '--data_path', default='/media/ssd_storage/HDM05/HDM05_mats')
    parser.add_argument(
        '--ignored_sample_path',
        default='')
    parser.add_argument('--out_folder', default='data/HDM05')

    arg = parser.parse_args()

    out_path = arg.out_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gendata(
        arg.data_path,
        out_path)
