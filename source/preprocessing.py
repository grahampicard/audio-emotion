import librosa
import os
import torch


def stft_transform(mp3path, filename='output', output_types=['stft', 'wave']):
    waveform, _ = librosa.load(mp3path)

    for output in output_types:
        if output == "stft":
            spec = abs(librosa.stft(waveform))
            spec = torch.Tensor(spec)            
            output_path = f"data/interim/stft/{filename}.pt"
            torch.save(spec, output_path)

        if output == "wave":
            wave = torch.Tensor(waveform)     
            output_path = f"data/interim/wave/{filename}.pt"
            torch.save(wave, output_path)

    return True


if __name__ == "__main__":

    mp3dir = 'data/raw/CAL500_32kps/'
    mp3files = [(mp3dir + f, f) for f in os.listdir(mp3dir)]

    for filepath, filename in mp3files:
        stft_transform(filepath, filename)