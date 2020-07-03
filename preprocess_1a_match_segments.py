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
    cols_file = './data/raw/CAL500_noAudioFeatures/vocab.txt'
    annot_path = './data/raw/SegLabelHard/SegLabelHard'

    with open(cols_file, 'r') as f: 
        cols = f.read().splitlines()    
        emotion_cols = [col for col in cols if 'Emotion-' in col if not 'NOT-' in col]

    annot_names = os.listdir(annot_path)

    # create lookup dataframe
    annotations = pd.concat([load_file(annot_path, f) for f in annot_names])
    annotations['duration'] = annotations['End_Time'] - annotations['Start_Time']
    annotations = annotations[['source', 'index', 'Start_Time', 'duration'] + emotion_cols]
    annotations.columns = [x.replace('Emotion-', '').replace('_/_', '_').lower() for x in annotations.columns]    
    annotations.to_csv('./data/raw/song_segment_times.csv', index=False)
