import librosa
import os
import torch


def simple_transformer(mp3path, savedirectory='./data/interim/features/',
                       filename='output',
                       transforms=['stft', 'wave', 'logmel', 'cqt'],
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
        if output == "wave":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            wave = torch.Tensor(waveform)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(wave, output_path)

        if output == "stft":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            spec = librosa.stft(waveform)
            spec_db = librosa.amplitude_to_db(abs(spec))
            spec_db = torch.Tensor(spec_db)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(spec_db, output_path)

        if output == "logmel":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
            mel = mel.astype(np.float16)
            logmel = np.log(10000 * mel + 1)
            logmel = torch.Tensor(logmel)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(logmel, output_path)

        if output == "cqt":
            pass
            #c = librosa.cqt(y=waveform, sr=sample_rate)

        if output == "chroma":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)
            

            harmonic,_ = librosa.effects.hpss(waveform)
            chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate,
                                                bins_per_octave=36)

            form = torch.Tensor(chroma)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(form, output_path)

    return True
