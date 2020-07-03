import argparse
import os
import pandas as pd
from source.preprocessing_audio import simple_transformer


if __name__ == "__main__":

    lengths = (5, 15, 30)

    for sample_length in lengths:
        
        # Audio
        savepath = f'./data/interim/{sample_length}secondsamples'
        mp3dir = './data/raw/CAL500_32kps/'
        mp3files = [(mp3dir + f, f.replace('.mp3', '')) for f in os.listdir(mp3dir)]
    
        for filepath, filename in mp3files:
            simple_transformer(filepath, savepath, filename,
                            transforms=['stft'], seconds=sample_length)

        # Labels
        idx_file = './data/raw/CAL500_noAudioFeatures/songNames.txt'
        cols_file = './data/raw/CAL500_noAudioFeatures/vocab.txt'
        data_file = './data/raw/CAL500_noAudioFeatures/hardAnnotations.txt'

        with open(idx_file, 'r') as f: idx = f.read().splitlines()    
        with open(cols_file, 'r') as f: cols = f.read().splitlines()

        # subset of columns with positively tagged emotions
        emotion_cols = [col for col in cols if 'Emotion-' in col 
                                            if not 'NOT-' in col]

        if not os.path.exists(f'{savepath}/labels/'):
            os.makedirs(f'{savepath}/labels/')

        # Convert to dataframe for easy labeling
        df = pd.read_csv(data_file, header=None)
        df.columns = cols
        df.index = idx

        # clean and label accordingly
        df_emotions = df[emotion_cols]
        df_emotions.columns = [x.replace('Emotion-', '')
                                .replace('_/_', '_')
                                .lower() for x in df_emotions.columns]
        df_emotions.index.name = 'song'
        df_emotions = df_emotions.reset_index()
        df_emotions.to_csv(f'{savepath}/labels/multi_label_emotions.csv',
                        index=False)
        
        # one-hot matrix for top emotion only
        df_emotions_melt = df_emotions.melt(id_vars=['song'], var_name='emotion')
        top_emotion = (df_emotions_melt.sort_values('value', ascending=False)
                                    .groupby('song')
                                    .first()
                                    .reset_index()
                                    [['song', 'emotion']])

        one_hot_top_emotion = pd.concat([top_emotion[['song']], 
                                        pd.get_dummies(top_emotion.emotion)],
                                        axis=1)

        one_hot_top_emotion.to_csv(f'{savepath}/labels/one_hot_top_emotion.csv',
                                index=False)
