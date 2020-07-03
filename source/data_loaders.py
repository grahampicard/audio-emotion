import numpy as np
import os
import pandas as pd
import torch


def load_15sec_stft_data(split=0.8, seed=123, label_type='one-hot'):

    # load files
    stft_dir = './data/interim/15secondsamples/stft'
    tensor_files = os.listdir(stft_dir)

    # add options for labels to use!
    if label_type == 'soft': csv_file = 'multi_label_emotions'
    if label_type == 'one-hot': csv_file = 'one_hot_top_emotion'
    label_df = pd.read_csv(f'./data/interim/15secondsamples/labels/{csv_file}.csv', index_col='song')

    # add features
    features = []
    labels = []

    # Load data
    for f in tensor_files:
        song = f.replace('.pt', '')
        cur_file = os.path.join(stft_dir, f)
        cur_song = torch.load(cur_file)
        cur_label = label_df.loc[song].to_numpy()

        if cur_song.shape[1] == 938:
            features.append(cur_song)
            labels.append(cur_label)

    # create train & test splits
    size = len(features)
    idxs = list(range(size))
    split_idx = int(np.floor(split * size))

    np.random.seed(seed)
    np.random.shuffle(idxs)

    train_split = idxs[:split_idx]
    test_split = idxs[split_idx:]
    
    features = torch.stack(features).unsqueeze(1)
    labels = torch.FloatTensor(labels)

    features_train, labels_train = features[train_split], labels[train_split]
    features_test, labels_test = features[test_split], labels[test_split]

    return features_train, labels_train, features_test, labels_test
