import streamlit as st
import librosa
import numpy as np
import pickle
import pyaudio
import wave
from io import BytesIO

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Audio Sentiment Prediction")

def process_audio(audio_data, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    max_time_steps = 128
    if mel_spec_db.shape[1] < max_time_steps:
        pad_width = max_time_steps - mel_spec_db.shape[1]
        mel_spec_padded = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_padded = mel_spec_db[:, :max_time_steps]

    # Make prediction
    input_data = np.array([mel_spec_padded])
    prediction = model.predict(input_data)
    y_pred = np.argmax(prediction, axis=1)

    return y_pred

def record_audio(recording_time, sample_rate):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    for _ in range(int(sample_rate / 1024 * recording_time)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_data = b''.join(frames)
    audio_file = BytesIO(audio_data)
    
    # Save as WAV
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    
    audio_file.seek(0)
    return audio_file

tab1, tab2 = st.tabs(["Upload File", "Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
        y_pred = process_audio(audio_data, sample_rate)
        class_labels = ["birds", "cats", "dogs"]
        st.write(f"You Said : {class_labels[y_pred[0]]}")

with tab2:
    recording_time = 2  # seconds
    sample_rate = 16000
    if st.button("Record"):
        audio_file = record_audio(recording_time, sample_rate)
        audio_data, _ = librosa.load(audio_file, sr=sample_rate)
        
        # Play the recorded audio
        st.audio(audio_file, format='audio/wav', sample_rate=sample_rate)
        
        y_pred = process_audio(audio_data, sample_rate)
        class_labels = ["birds", "cats", "dogs"]
        st.write(f"You Said : {class_labels[y_pred[0]]}")
