import os
import librosa
import numpy as np
import pandas as pd
from zenml import step

@step
def ingest_data() -> tuple:
    data_path = r'C:\Users\Parth\Documents\Audio_Sentiment\data\Animals'
    def load_and_extract_spectrogram(data_path, n_mels=128, n_fft=2048, hop_length=512):
        y, sr = librosa.load(file_path, sr=None)  
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  
        return mel_spec_db
    subdirectories = [subdir for subdir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, subdir))]
    X = []
    y = []
    max_time_steps = 128  
    for idx, subdir in enumerate(subdirectories):
        subdir_path = os.path.join(data_path, subdir)
        wav_files = [file for file in os.listdir(subdir_path) if file.endswith('.wav')]

        for wav_file in wav_files:
            file_path = os.path.join(subdir_path, wav_file)
            spectrogram = load_and_extract_spectrogram(file_path)
            if spectrogram.shape[1] < max_time_steps:
                pad_width = max_time_steps - spectrogram.shape[1]
                spectrogram_padded = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                X.append(spectrogram_padded)
            else:
                X.append(spectrogram[:, :max_time_steps])  # Trim if spectrogram has more time steps
            y.append(idx)  # Use class index as label
    return (X,y)