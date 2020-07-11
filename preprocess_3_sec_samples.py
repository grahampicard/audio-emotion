import os
import pandas as pd
from source.preprocessing_audio import simple_transformer


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


    # Audio
    ## get files
    mp3dir = './data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f,
                 f.replace('.mp3', '')) for f in os.listdir(mp3dir)]
    mp3df = pd.DataFrame(mp3files, columns=['path', 'source'])

    ## find durations
    cols = ['source', 'index', 'start_time', 'end_time']
    segments = pd.read_csv('./data/raw/song_segment_times.csv')
    segments['end_time'] = segments['start_time'] + segments['duration']

    matches = segments.merge(mp3df, on='source')
    matches = matches[cols].to_records(index=False)
    segment_path = 'data/interim/expanded-3secondsegments/'
    taggable_segments = []

    if not os.path.exists(segment_path): os.makedirs(segment_path)
    for source, order, start, end in matches:

        clip_start = start
        clip_end = clip_start + 3

        while clip_end <= end:
            try:
                mp3path = os.path.join(mp3dir, source) + '.mp3'
                filename = f'{source}-seg-{order}-time-{int(clip_start)}-{int(clip_end)}'
                simple_transformer(mp3path, segment_path, filename, transforms=['stft'], seconds=3, offset=clip_start)
                taggable_segments.append((source, order, start, end))
            except:
                pass
            finally:
                clip_start += 3
                clip_end += 3

    ## Make target path
    labels = pd.DataFrame(taggable_segments, columns=cols)
    labels = labels.merge(segments).drop(columns=['start_time', 'duration'])
    labels = labels.set_index(['source', 'index'])

    label_path = os.path.join(segment_path, 'labels/')
    if not os.path.exists(label_path): os.makedirs(label_path)
    segments.to_csv(os.path.join(segment_path,'labels/multi_label_emotions.csv'))
