import librosa
import os
import torch


def simple_stft_transform(mp3path, filename='output', output_types=['stft', 'wave'], sample_rate=32000, seconds=15):
    """ Simple loader which will take a path/to/mp3 and convert it to numerical waveform files.

        Parameters
        ----------------

        mp3path         path to input file
        filename        stem which will be used to create `.pt` files
        output_types    transformations to be applied to file 
        sample_rate     number of cycles measured per second
        seconds         duration of the clip
    """

    if isinstance(output_types, str):
        output_types = [output_types]

    waveform, _ = librosa.load(mp3path, sr=sample_rate, duration=seconds)

    for output in output_types:
        if output == "stft":    
            spec = abs(librosa.stft(waveform))
            spec = torch.Tensor(spec)

            if spec.shape[1] == 938:
                output_path = f"./../data/interim/features/stft/{filename}.pt"
                torch.save(spec, output_path)

        if output == "wave":
            wave = torch.Tensor(waveform)     
            if wave.shape[1] == 938:
                output_path = f"./../data/interim/features/wave/{filename}.pt"
                torch.save(wave, output_path)

    return True
    

if __name__ == "__main__":

    mp3dir = './../data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f, f.replace('.mp3', '')) for f in os.listdir(mp3dir)]

    for filepath, filename in mp3files:
        simple_stft_transform(filepath, filename, output_types=['stft'])
