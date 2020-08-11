import json
import os
import pandas as pd


if __name__ == "__main__":
    
    with open('data/raw/spotify_features.json', 'r') as f:
        data = json.load(f)

        cols = ['cal_id', 'danceability', 'energy', 'key', 'loudness',
                'mode', 'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'time_signature']

        df = pd.DataFrame(data)[cols]

    dir_path = "./data/interim/spotify"
    if not os.path.exists(dir_path): os.makedirs(dir_path)

    df.to_csv(f'{dir_path}/spotify_features.csv', index=False)


    # create labels file
    cols_file = './data/raw/CAL500_noAudioFeatures/vocab.txt'  
    with open(cols_file, 'r') as f: 
        cols = f.read().splitlines()    
        emotion_cols = [col for col in cols if 'Emotion-' in col if not 'NOT-' in col]

    row_names = pd.read_csv('./data/raw/CAL500_noAudioFeatures/songNames.txt', sep='\t')
    row_names = row_names.to_numpy().reshape(-1,)
    labels = pd.read_csv('./data/raw/CAL500_noAudioFeatures/hardAnnotations.txt')
    labels.index = row_names
    labels.columns = cols
    labels = labels[emotion_cols]
    labels.columns = [x.replace('Emotion-', '').replace('_/_', '_').lower() for x in labels.columns]
    labels.to_csv(f'{dir_path}/labels.csv')