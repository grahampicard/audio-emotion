import numpy as np
import os
import pandas as pd
import torch


def load_stft_data(valid_split=0.8, test_split=0.9, seed=None,
                   label_type='one-hot', sample_length=15):

    # do manual checks for segment. found by observing output from librosa
    if sample_length == 15:
        expected_shape = 938
    elif sample_length == 30:
        expected_shape = 1876
    else:
        raise ValueError

    # load files
    stft_dir = f'./data/interim/{sample_length}secondsamples/stft'
    tensor_files = os.listdir(stft_dir)

    # add options for labels to use!
    if label_type == 'soft': csv_file = 'multi_label_emotions'
    if label_type == 'one-hot': csv_file = 'one_hot_top_emotion'
    label_df = pd.read_csv(f'./data/interim/{sample_length}secondsamples/labels/{csv_file}.csv', index_col='song')

    # add features
    features = []
    labels = []

    # Load data
    for f in tensor_files:
        song = f.replace('.pt', '')
        cur_file = os.path.join(stft_dir, f)
        cur_song = torch.load(cur_file)
        cur_label = label_df.loc[song].to_numpy()

        # Ensure all samples are 30 seconds long
        if cur_song.shape[1] == expected_shape:
            features.append(cur_song)
            labels.append(cur_label)

    # create train & test splits
    size = len(features)
    idxs = list(range(size))

    if seed is not None: 
        np.random.seed(seed)
        
    np.random.shuffle(idxs)

    train_idx, valid_idx, test_idx = np.split(idxs, [int(valid_split * size), int(test_split * size)])

    features = torch.stack(features).unsqueeze(1)
    labels = torch.FloatTensor(labels)

    features_train, labels_train = features[train_idx], labels[train_idx]
    features_valid, labels_valid = features[valid_idx], labels[valid_idx]
    features_test, labels_test = features[test_idx], labels[test_idx]

    return features_train, labels_train, features_valid, labels_valid, features_test, labels_test
