import json
import pandas as pd


if __name__ == "__main__":
    
    with open('data/raw/spotify_features.json', 'r') as f:
        data = json.load(f)

        cols = ['cal_id', 'danceability', 'energy', 'key', 'loudness',
                'mode', 'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'time_signature']

        df = pd.DataFrame(data)[cols]
        df.to_csv('./data/interim/metadata/spotify_features.csv', index=False)