import numpy as np
import pandas as pd
import os

import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_stft():

    error_msg = "Create a directory `stft` with tensor files for each song"
    stft_dir = './../data/interim/features/stft'
    assert os.path.isdir(stft_dir), error_msg

    # load files
    tensor_files = os.listdir(stft_dir)
    label_df = pd.read_csv('./../data/interim/labels/emotional_scores.csv', index_col='song')

    features = []
    labels = []

    for f in tensor_files:
        song = f.replace('.pt', '')
        cur_file = os.path.join(stft_dir, f)
        features.append(torch.load(cur_file))
        
        cur_label = label_df.loc[song].to_numpy()
        labels.append(cur_label)

    features = torch.stack(features)
    labels = torch.FloatTensor(labels)

    return features, labels
    

if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = load_stft()

    # data_size = len(features)
    # indices = list(range(data_size))
    # split = int(np.floor(0.2 * data_size))

    # np.random.seed(123)
    # np.random.shuffle(indices)

    # train_idx, test_idx = indices[split:], indices[:split]
    # train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
