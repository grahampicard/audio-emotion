# Getting started
Because of size limitations, we can't include all our audio files in the github repo. In order for this to work, you'll need to first download all of the raw files (e.g. `CAL500_32kps`) that are stored in our box.com account under `CAL500 Data (drop this into data-raw of recreation paper)`. Then, run files accordingly. 

# Folder Structure

```
|---data
|    |---   raw                         extracts from CAL500 dataset
|    |---   interim                     pre-processed data
|    |---   final                       processed data / data for sharing
|
|---models                              Models that will be trained
|    |---   simple_cnn          
|
|---source                              Code for handling all development
    |---    preprocessing_data.py       handles all audio data preprocessing
    |---    preprocessing_labels.py     cleans and creates labels file
    |---    train_simple_cnn.py         
```

# Getting Started
1. Downlod `raw - (unzip into a folder named raw).zip` from our box.com account
2. Unzip this add all contents into the folder named `data/raw`. Resulting file structure should look like this:
        
        data/raw/CA500_32kps
        data/raw/DeltaMFCCFeatures
        ...

