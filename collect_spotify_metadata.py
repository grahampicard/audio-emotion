from source.metadata_handler import track_metadata_handler
from time import sleep

import json
import pandas as pd


if __name__ == "__main__":
    
    handler = track_metadata_handler()

    tracks = pd.read_csv('./data/raw/cal500_EchoNest_IDs.txt', sep='\t', header=None)
    tracks.columns = 'artisttrack', 'echonestid'
    tracks['clean_label'] = tracks['artisttrack'].str.replace('_', ' ')
    
    search_params = (tracks['clean_label'].str.split('-', 1, expand=True))
    search_params.loc[66] = '', 'bjork army of me'
    search_params.loc[74] = '', 'blue oyster cult burnin for you'
    search_params.loc[94] = '', 'brenton wood lovey dovey'
    search_params.loc[125] = '', 'cc music factory gonna make you sweat'
    search_params.loc[135] = '', 'chi-lites stoned out of my mind'
    search_params.loc[308] = '', 'michael masley adive from the angel of thresholds'
    search_params.loc[327] = '', 'neil young western hero'
    search_params.loc[351] = '', 'pizzle what\'s wrong with my foot'

    search_params = search_params.to_records()

    results = []

    print("Skipping...")

    for row in search_params:
        idx, artist, track = row
        idx = int(idx)

        try:
            resp = handler.get_track_metadata(artist=artist, track=track,
                                              spotify=True, lastfm=False)   
            resp['index'] = idx
            resp['cal_id'] = tracks.loc[idx, 'artisttrack']
            results.append(resp)
        except:
            print(idx, artist, track)

    with open('data/interim/spotify/spotify_features.json', 'w') as f:
        json.dump(results, f, indent='  ')
