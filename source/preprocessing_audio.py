import librosa
import os
import torch


def simple_transformer(mp3path, savedirectory='./data/interim/features/',
                       filename='output', transforms=['stft', 'wave'],
                       sample_rate=32000, seconds=30, offset=0.0):

    """ Simple loader which will take a path/to/mp3 and convert it to 
        numerical waveform files.

        Parameters
        ----------------

        mp3path         path to input file e.g.             `/data/raw/f.mp3`
        savedirectory   directory where output is stored    `/interim/outputs/`
        filename        stem used to create `.pt` files     `sample-output`
        transforms      transformations applied to audio    `['stft']`
        sample_rate     number of cycles sampled per second `22050`
        seconds         duration of the clip                `30`
        offset          start time duration                 `0.0`
    """

    if isinstance(transforms, str): transforms = [transforms]

    # load librosa file
    waveform, _ = librosa.load(mp3path, sr=sample_rate, duration=seconds,
                               offset=offset)

    # add transforms here
    for output in transforms:
        if output == "stft":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            spec = abs(librosa.stft(waveform))
            spec = torch.Tensor(spec)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(spec, output_path)

        if output == "wave":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            wave = torch.Tensor(waveform)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(wave, output_path)

        if output == "chroma":
            pass

    return True
