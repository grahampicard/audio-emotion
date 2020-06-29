import os
import pandas as pd
from source.preprocessing_audio import simple_stft_transform


if __name__ == "__main__":

    # Audio
    ## get files
    mp3dir = './data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f, f.replace('.mp3', '')) for f in os.listdir(mp3dir)]
    mp3df = pd.DataFrame(mp3files, columns=['path', 'source'])

    ## find durations
    segments = pd.read_csv('./data/raw/song_segment_times.csv')
    matches = segments.merge(mp3df, on='source')
    matches = matches[['source', 'order', 'Start_Time', 'segment_duration']].to_records(index=False)
    
    ## Make target path
    segment_path = 'data/interim/segments/features/stft/'
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)

    for source, order, start, dur in matches:
        mp3path = os.path.join(mp3dir, source) + '.mp3'
        filename = source + '-' + str(order)
        simple_stft_transform(mp3path, segment_path, filename, output_types=['stft'], seconds=dur)
