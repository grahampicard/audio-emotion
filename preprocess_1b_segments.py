import os
import pandas as pd
from source.preprocessing_audio import simple_transformer


if __name__ == "__main__":

    # Audio
    ## get files
    mp3dir = './data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f,
                 f.replace('.mp3', '')) for f in os.listdir(mp3dir)]
    mp3df = pd.DataFrame(mp3files, columns=['path', 'source'])

    ## find durations
    match_cols = ['source', 'index', 'start_time', 'duration']
    segments = pd.read_csv('./data/raw/song_segment_times.csv')
    matches = segments.merge(mp3df, on='source')
    matches = matches[match_cols].to_records(index=False)
    
    ## Make target path
    segment_path = 'data/interim/variable-segments/'
    taggable_segments = []
    if not os.path.exists(segment_path): os.makedirs(segment_path)
    for source, order, start, dur in matches:
        try:
            mp3path = os.path.join(mp3dir, source) + '.mp3'
            filename = source + '-' + str(order)
            simple_transformer(mp3path, segment_path, filename,
                               transforms=['stft'], seconds=dur)
            taggable_segments.append((source, order, start, dur))
        except:
            print(f"Skipping {source} - {order}")

    # Labels
    ## open files
    labels = pd.DataFrame(taggable_segments, columns=match_cols)
    labels = labels.merge(segments).drop(columns=['start_time', 'duration'])
    labels = labels.set_index(['source', 'index'])

    label_path = os.path.join(segment_path, 'labels/')
    if not os.path.exists(label_path): os.makedirs(label_path)
    labels.to_csv(os.path.join(segment_path,'labels/multi_label_emotions.csv'))
