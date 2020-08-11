import librosa
import os
import pandas as pd
import subprocess


def load_file(parent_dir, file_name):
    """ Loads file into a dataframe """
    df = pd.read_csv(os.path.join(parent_dir, file_name))
    df['source'] = file_name.replace('.csv', '')
    df = df.reset_index()
    return df


if __name__ == "__main__":
    """
    Take MP3 audio files and apply preprocessing functions to run a 
    short-time Fourier transform on each MP3 file. 

    This script collects path names for MP3s and outputs `.pt` files
    with STFT to view audio frequencies. 

    Folder conventions

    Input:
        data
          |__raw
            |__name-of-mp3-files

    Output:
        data
          |__interim
            |__name-of-sample-length
              |__transformation
              |__transformation2
              ...
              |__labels
    """

    cols_file = './data/raw/CAL500_noAudioFeatures/vocab.txt'
    hard_annot_path = './data/raw/SegLabelHard/SegLabelHard'
    soft_annot_path = './data/raw/SegLabelSoft/SegLabelSoft'    

    with open(cols_file, 'r') as f: 
        cols = f.read().splitlines()    
        emotion_cols = [col for col in cols if 'Emotion-' in col if not 'NOT-' in col]

    hard_annot_names = os.listdir(hard_annot_path)
    soft_annot_names = os.listdir(soft_annot_path)

    # create lookup dataframe
    annotations_hard = pd.concat([load_file(hard_annot_path, f) for f in hard_annot_names])
    annotations_hard['duration'] = annotations_hard['End_Time'] - annotations_hard['Start_Time']
    annotations_hard = annotations_hard[['source', 'index', 'Start_Time', 'duration'] + emotion_cols]
    annotations_hard.columns = [x.replace('Emotion-', '').replace('_/_', '_').lower() for x in annotations_hard.columns]    
    annotations_hard.to_csv('./data/raw/song_segment_times_hard.csv', index=False)

    annotations_soft = pd.concat([load_file(soft_annot_path, f) for f in soft_annot_names])
    annotations_soft['duration'] = annotations_soft['End_Time'] - annotations_soft['Start_Time']
    annotations_soft = annotations_soft[['source', 'index', 'Start_Time', 'duration'] + emotion_cols]
    annotations_soft.columns = [x.replace('Emotion-', '').replace('_/_', '_').lower() for x in annotations_soft.columns]
    annotations_soft.to_csv('./data/raw/song_segment_times_soft.csv', index=False)
