import librosa
import numpy as np
import os
import sklearn
import torch


def simple_transformer(mp3path, savedirectory='./data/interim/features/',
                       filename='output',
                       transforms=['stft', 'wave', 'logmel', 'mfcc', 'chroma',
                       'cqt'],
                       sample_rate=32000, seconds=30, offset=0.0):

    """ Simple loader which will take a path/to/mp3 and convert it a selected
        transform file.

        Parameters
        ----------------
        mp3path         path to input file e.g.             `/data/raw/f.mp3`
        savedirectory   directory where output is stored    `/interim/outputs/`
        filename        stem used to create `.pt` files     `sample-output`
        transforms      transformations applied to audio    `['stft']`
        sample_rate     number of cycles sampled per second `22050`
        seconds         duration of the clip                `30`
        offset          start time duration                 `0.0`


        Transforms
        ---------------
        STFT            Short-Time Fourier Transform (Spectrogram)
        Wave            Waveform (no transform)
        Log-Mel         Logged values of the Mel Spectrogram
        MFCC            Mel-Frequency Cepstral Coeffcient
        Chroma          Chroma
        CQT             Constant Q Transform
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

        elif output == "stft":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            spec = librosa.stft(waveform)
            spec_db = librosa.amplitude_to_db(abs(spec))
            spec_db = torch.Tensor(spec_db)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(spec_db, output_path)

        elif output == "logmel":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
            mel = mel.astype(np.float16)
            logmel = np.log(10000 * mel + 1)
            logmel_db = librosa.amplitude_to_db(abs(logmel))
            logmel_db = torch.Tensor(logmel_db)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(logmel_db, output_path)

        elif output == "chroma":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            harmonic,_ = librosa.effects.hpss(waveform)
            chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate,
                                                bins_per_octave=36) # chroma_stft???
            form = torch.Tensor(chroma)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(form, output_path)

        elif output == "mfcc":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            mfccs = librosa.feature.mfcc(waveform, sr=sample_rate)
            mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
            mfcc_tensor = torch.Tensor(mfccs)

            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(mfcc_tensor, output_path)

        elif output == "cqt":
            dir_path = os.path.join(savedirectory, output)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            c = librosa.cqt(y=waveform, sr=sample_rate, bins_per_octave=36)
            c_db = librosa.amplitude_to_db(abs(c))
            c_db = torch.Tensor(c_db)
            output_path = os.path.join(dir_path, f'{filename}.pt')
            torch.save(c_db, output_path)

        else:
            raise ValueError("Enter a valid transform")

    return True
