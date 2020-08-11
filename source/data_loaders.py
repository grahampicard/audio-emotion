import numpy as np
import os
import pandas as pd
import torch


def load_section_level_stft(valid_split=0.8, test_split=0.9, seed=None,
                            label_type='soft-labels', dev=False, preprocessing='stft'):
    """
    Uses CAL500 Expanded data where there are varaible length segments with
    different emotions for each segments. 
    Assumptions:
    1.  don't allow overlap between songs and testing split
    2.  each song is 3 seconds long
    3.  there is only one label record (soft/hard/one-hot) per song
    """

    if seed is not None: 
        np.random.seed(seed)

    # do manual checks for segment. found by observing output from librosa
    if preprocessing == "wave":
        shape_idx = 0
        expected_shape = 96000
    else:
        shape_idx = 1
        expected_shape = 188

    # load files
    data_dir = f'./data/interim/expanded-3secondsegments/{preprocessing}'
    tensor_files = os.listdir(data_dir)

    if dev:
        tensor_files = tensor_files[:len(tensor_files) // 10]

    # find all songs and divide into train, test, validate
    all_songs = [x.split('-seg-')[0] for x in tensor_files]
    segment_counter = {song: 0 for song in set(all_songs)}
    for song in all_songs:
        segment_counter[song] += 1

    # create train & test splits
    size = len(all_songs)
    n_test = int(size * (1 - test_split))
    n_valid = int(size * (test_split - valid_split))

    # shuffle keys
    keys = list(segment_counter.keys())
    np.random.shuffle(keys)
    valid_keys, valid_counter = [], 0
    test_keys, test_counter = [], 0
    train_keys = []

    for key in keys:

        if valid_counter < n_valid:
            valid_keys.append(key)
            valid_counter += segment_counter[key]
        elif test_counter < n_test:
            test_keys.append(key)
            test_counter += segment_counter[key]
        else:
            train_keys.append(key)
       
    # add options for labels to use!
    csv_file = f'./data/interim/expanded-3secondsegments/labels/{label_type}.csv'

    if not os.path.exists(csv_file):
        raise ValueError
    
    label_df = pd.read_csv(csv_file, index_col=['source', 'index'])

    # get features and labels
    features_train, labels_train, = [], [] 
    features_valid, labels_valid, = [], [] 
    features_test, labels_test, = [], []

    # Load data
    for f in tensor_files:
        song, idx = f.split('-seg-')
        idx = int(idx.split('-time-')[0])
        cur_file = os.path.join(data_dir, f)
        cur_feature = torch.load(cur_file)

        if (song, idx) in label_df.index:

            cur_label = label_df.loc[(song, idx)].to_numpy()

            # Ensure all samples are 3 seconds long
            if cur_feature.shape[shape_idx] == expected_shape:
                if song in train_keys:
                    features_train.append(cur_feature)
                    labels_train.append(cur_label)

                elif song in test_keys:
                    features_test.append(cur_feature)
                    labels_test.append(cur_label)

                elif song in valid_keys:
                    features_valid.append(cur_feature)
                    labels_valid.append(cur_label)    
            else:
                print(f)
    
    features_train, labels_train = torch.stack(features_train).unsqueeze(1), torch.FloatTensor(labels_train)
    features_test, labels_test = torch.stack(features_test).unsqueeze(1), torch.FloatTensor(labels_test)
    features_valid, labels_valid = torch.stack(features_valid).unsqueeze(1), torch.FloatTensor(labels_valid)

    return features_train, labels_train, features_test, labels_test, train_split, test_split


def load_spotify_metadata(valid_split=0.8, test_split=0.9, seed=123, csv_file='one_hot_top_emotion'):

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

    train_idx = idxs[:valid_split]
    valid_idx = idxs[valid_split:test_split]
    test_idx = idxs[test_split;]
    
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    features_train, labels_train = features[train_idx], labels[train_idx]
    features_test, labels_test = features[test_idx], labels[test_idx]
    features_valid, labels_valid = features[valid_idx], labels[valid_idx]

    return features_train, labels_train, features_test, labels_test, features_valid, labels_valid