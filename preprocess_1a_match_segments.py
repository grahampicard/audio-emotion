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

def duration(mp3_path, mp3_name):
    """ Uses FFMPEG to find audio length. Faster than librosa. """
    args=("ffprobe","-show_entries", "format=duration","-i",
          os.path.join(mp3_path, mp3_name))
    popen = subprocess.Popen(args, stdout = subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    _, length = output.decode().split('\r\n')[1].split('=')
    return (mp3_name.replace('.mp3', ''), float(length))


if __name__ == "__main__":
    cols_file = './data/raw/CAL500_noAudioFeatures/vocab.txt'
    with open(cols_file, 'r') as f: cols = f.read().splitlines()    
    emotion_cols = [col for col in cols if 'Emotion-' in col if not 'NOT-' in col]

    annot_path = './data/raw/SegLabelHard/SegLabelHard'
    annot_names = os.listdir(annot_path)
    annotations = pd.concat([load_file(annot_path, f) for f in annot_names])
    annotations['duration'] = annotations['End_Time'] - annotations['Start_Time']
    annotations = annotations[['source', 'index', 'Start_Time', 'duration'] + emotion_cols]
    annotations.columns = [x.replace('Emotion-', '').replace('_/_', '_').lower() for x in annotations.columns]    
    annotations.to_csv('./data/raw/song_segment_times.csv', index=False)
