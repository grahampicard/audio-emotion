import os
import pandas as pd


hard_labels = pd.read_csv('data/interim/expanded-3secondsegments/labels/hard-labels.csv', index_col=[0,1])

for label in hard_labels.columns:
    new_df = hard_labels[[label]] 
    new_df.to_csv(f'data/interim/expanded-3secondsegments/labels/{label}.csv')