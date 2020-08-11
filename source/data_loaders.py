import numpy as np
import os
import pandas as pd
import torch


def load_section_level_stft(valid_split=0.8, test_split=0.9, seed=None,
                            label_type='soft-labels', dev=False, preprocessing='stft'):
    """
    Parameters
    ----------------
    valid_split     point on interval [0,1] to split train/valid    0.8
    test_split      point on interval [0,1] to split valid/test     0.9
    seed            use for reproducable results                    123
    label_type      either using hard or soft labels                'soft-labels'
                    'hard-labels': multilabel
                    'soft-labels': probability of emotion occurring
                                   (assuming binary for each emotion)
                    ex: hard:    0    0    1    1
                        soft:    0    0    0.67 1.0
    dev             if True then use smaller (10%) amount of data  False
    preprocessing   'stft', 'wave', 'logmel', 'mfcc', 'chroma',    'stft'
                    'cqt'

    Uses CAL500 Expanded data where there are variable length segments with
    different emotions for each segment. Splits data into train, valid, and test
    so that all segments of a song fall within the same bucket (to avoid data
    leakage).

    Assumptions:
    1.  don't allow overlap between songs and testing split
    2.  each song is 3 seconds long
    3.  there is only one label record (soft/hard/one-hot) per song
    """

    if seed is not None:
        np.random.seed(seed)

    # found by observing output from librosa
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

    # create train, valid, & test splits
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

    return features_train, labels_train, features_valid, labels_valid, features_test, labels_test
