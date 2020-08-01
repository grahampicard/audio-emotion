import os
import pandas as pd


hard_labels = pd.read_csv('data/interim/expanded-3secondsegments/labels/hard-labels.csv', index_col=[0,1])

# creates a 1/0 classification for a given emotion ('e.g. 1=HAPPY or 0=NOT HAPPY')
for label in hard_labels.columns:
    new_df = hard_labels[[label]] 
    new_df.to_csv(f'data/interim/expanded-3secondsegments/labels/{label}.csv')

# creates a 1/0 classication for an emotion against "happy" ('e.g. 1=HAPPY or 0=SAD')
for label in hard_labels.columns:
    if label != 'happy':
        new_df = hard_labels[['happy', label]] 
        new_df = new_df.loc[new_df['happy'] != new_df[label]][['happy']]
        new_df.to_csv(f'data/interim/expanded-3secondsegments/labels/happy-not-{label}.csv')
