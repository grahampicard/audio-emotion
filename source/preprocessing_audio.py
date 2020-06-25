import librosa
import os
import torch


def simple_stft_transform(mp3path, savepath='./data/interim/', filename='output', output_types=['stft', 'wave'], sample_rate=32000, seconds=30):
    """ Simple loader which will take a path/to/mp3 and convert it to numerical waveform files.

        Parameters
        ----------------

        mp3path         path to input file
        savepath        path to save output
        filename        stem which will be used to create `.pt` files
        output_types    transformations to be applied to file 
        sample_rate     number of cycles measured per second
        seconds         duration of the clip
    """

    if isinstance(output_types, str):
        output_types = [output_types]

    waveform, _ = librosa.load(mp3path, sr=sample_rate, duration=seconds)

    if waveform.shape[0] != sample_rate * seconds:
        return False

    for output in output_types:
        if output == "stft":    
            spec = abs(librosa.stft(waveform))
            spec = torch.Tensor(spec)
            output_path = os.path.join(savepath, 'stft/', f'{filename}.pt')
            torch.save(spec, output_path)

        if output == "wave":
            wave = torch.Tensor(waveform)     
            output_path = os.path.join(savepath, 'wave/', f'{filename}.pt')
            torch.save(wave, output_path)

    return True