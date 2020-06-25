import numpy as np
import os
import pandas as pd
import torch


def load_stft_data(split=0.8, seed=123, csv_file='one_hot_top_emotion'):

    # load files
    stft_dir = './data/interim/features/stft'
    tensor_files = os.listdir(stft_dir)

    # add options for labels to use!  
    label_df = pd.read_csv(f'./data/interim/labels/{csv_file}.csv', index_col='song')

    # add features
    features = []
    labels = []

    # Load data
    for f in tensor_files:
        song = f.replace('.pt', '')
        cur_file = os.path.join(stft_dir, f)
        features.append(torch.load(cur_file))
        
        cur_label = label_df.loc[song].to_numpy()
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

    return features_train, labels_train, features_test, labels_test, train_split, test_split


def load_spotify_metadata(split=0.8, seed=123, csv_file='one_hot_top_emotion'):

    # add options for labels to use!
    
    label_df = pd.read_csv(f'./data/interim/labels/{csv_file}.csv', index_col='song')
    feature_df = pd.read_csv('./data/interim/metadata/spotify_features.csv', index_col='cal_id')

    label_df = label_df.reindex(feature_df.index)

    assert label_df.shape[0] == feature_df.shape[0]

    # add features
    features = feature_df.to_numpy()
    labels = label_df.to_numpy()

    # create train & test splits
    size = len(features)
    idxs = list(range(size))
    split_idx = int(np.floor(split * size))

    np.random.seed(seed)
    np.random.shuffle(idxs)

    train_split = idxs[:split_idx]
    test_split = idxs[split_idx:]
    
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    features_train, labels_train = features[train_split], labels[train_split]
    features_test, labels_test = features[test_split], labels[test_split]

    return features_train, labels_train, features_test, labels_test, train_split, test_split
