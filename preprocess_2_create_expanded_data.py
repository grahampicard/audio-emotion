import argparse
import os
import pandas as pd
from source.preprocessing_audio import simple_transformer

if __name__ == "__main__":
<<<<<<< HEAD
    seconds = 3
    dev=False
    transforms=['wave', 'stft', 'logmel', 'chroma', 'mfcc']
    
    mp3dir = './data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f,
                 f.replace('.mp3', '')) for f in os.listdir(mp3dir)]
    mp3df = pd.DataFrame(mp3files, columns=['path', 'source'])

    ## find durations
    cols = ['source', 'index', 'start_time', 'end_time']
    segments_hard = pd.read_csv('./data/raw/song_segment_times_hard.csv')
    segments_hard['end_time'] = segments_hard['start_time'] + segments_hard['duration']

    segments_soft = pd.read_csv('./data/raw/song_segment_times_soft.csv')
    segments_soft['end_time'] = segments_soft['start_time'] + segments_soft['duration']

    matches = segments_hard.merge(mp3df, on='source')
    matches = matches[cols].to_records(index=False)
    segment_path = f'data/interim/expanded-{seconds}secondsegments/'
    if dev:
        segment_path = f'data/interim/expanded-dev{seconds}secondsegments/'
    tagged_segs = []

    if dev:
      matches = matches[:25]

    if not os.path.exists(segment_path): os.makedirs(segment_path)
    for source, order, start, end in matches:

        clip_start = start
        clip_end = clip_start + seconds

        while clip_end <= end:
            try:
                mp3path = os.path.join(mp3dir, source) + '.mp3'
                filename = f'{source}-seg-{order}-time-{int(clip_start)}-{int(clip_end)}'
                simple_transformer(mp3path, segment_path, filename,
                                   transforms=transforms, seconds=3,
                                   offset=clip_start)
                tagged_segs.append((source, order, start, end))
            except:
                pass
            finally:
                clip_start += seconds
                clip_end += seconds

    ## Make target path
    label_path = os.path.join(segment_path, 'labels/')
    if not os.path.exists(label_path): os.makedirs(label_path)

    segments_hard = segments_hard.drop(columns=['start_time', 'duration', 'end_time'])
    segments_hard.to_csv(os.path.join(segment_path,'labels/hard-labels.csv'), index=False)

    segments_soft = segments_soft.drop(columns=['start_time', 'duration', 'end_time'])
    segments_soft.to_csv(os.path.join(segment_path,'labels/soft-labels.csv'), index=False)
=======
    match_expanded_dataset(3, transforms=['logmel', 'mfcc', 'chroma'])
    
>>>>>>> master
