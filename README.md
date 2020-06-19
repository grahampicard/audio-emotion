# Getting Started
1. Downlod `raw - (unzip into a folder named raw).zip` from our box.com account
2. Unzip this add all contents into the folder named `data/raw`. Resulting file structure should look like this:
        
        data/raw/CA500_32kps
        data/raw/DeltaMFCCFeatures
        ...
3. Train model

        py preprocess.py
        py train_cnn_boilerplate.py


# Folder Structure
```
|---data
|   |---    raw                         extracts from CAL500 dataset
|   |---    interim                     pre-processed data
|   |---    processed                   processed data / data for sharing
|
|---models                              
|   |---    cnn_boilerplate             using CNN from github folder
|   |---    cnn_small                   smaller CNN for pipeline development
|
|---source                              Code for handling all development
|   |---    preprocessing_audio.py      handles all audio data preprocessing
|
|---preprocess.py                       Run to pre-process CAL500 Audio data
|---train_cnn_boilerplate.py            Training script to develop models
|---workspace.code-workspace            GP's workspace file
```
