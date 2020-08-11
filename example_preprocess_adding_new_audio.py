"""
In this script, Let's look at what it might take to add a new dataset. 
Let's assume that we've: (1) downloaded an MP3 dataset, (2) moved the 
dataset into the "data/raw" folder. I have moved a directory called 
"sample_music_data" into this folder which has Nils Frahm MP3s.

I also manually created a label file, which has a start and end time
for a song as well as a made up label "good".
"""

import argparse
import os
import pandas as pd
from source.preprocessing_audio import simple_transformer


if __name__ == "__main__":

    # declare transforms and point to the directory with the MP3 files
    transforms=['wave', 'stft', 'logmel', 'chroma', 'mfcc']    
    mp3dir = './data/raw/sample_music_data/'
    mp3files = [(mp3dir + f, f.replace('.mp3', '')) for f in os.listdir(mp3dir) 
                                                    if '.mp3' in f]
    mp3df = pd.DataFrame(mp3files, columns=['path', 'source'])

    # Load the label & time file. These could be separate. I chose to do one
    # for the sake of this example
    cols = ['source', 'index', 'start_time', 'end_time', 'good']
    segments = pd.read_csv(os.path.join(mp3dir, 'labels.csv'))
    matches = segments.merge(mp3df, on='source')
    matches = matches[cols].to_records(index=False)

    # the path where you want to save files
    segment_path = f'data/interim/example/'
    if not os.path.exists(segment_path): os.makedirs(segment_path)

    # let's loop through all matches and preprocess the song
    tagged_segs = []
    errors = []

    for source, order, start, end, label in matches:

        try:        
            duration = end - start

            mp3path = os.path.join(mp3dir, source) + '.mp3'
            filename = source
            simple_transformer(mp3path, segment_path, filename,
                                transforms=transforms, seconds=duration,
                                offset=start)
            tagged_segs.append((source, label))
        except:
            errors.append((source, order))
            """ If it doesn't work, we'll simply won't have a label
                record, and add an error record
            """
            pass

    ## Make target path
    label_path = os.path.join(segment_path, 'labels/')
    if not os.path.exists(label_path): os.makedirs(label_path)

    segments = segments.drop(columns=['start_time', 'end_time', 'index'])
    segments.to_csv(os.path.join(segment_path,'labels/labels.csv'), index=False)

    if errors:
        for error in error:
            print(f"{error} skipped")
