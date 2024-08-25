import streamlit as st
import librosa
import numpy as np
import pickle
import sounddevice as sd

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
        recording_samples = int(recording_time * sample_rate)
        recording = sd.rec(recording_samples, samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until the recording is finished
        audio_data = recording[:, 0]

        # Play the recorded audio
        st.audio(audio_data, format='audio/wav', sample_rate=sample_rate)
        
        y_pred = process_audio(audio_data, sample_rate)
        class_labels = ["birds", "cats", "dogs"]
        st.write(f"You Said : {class_labels[y_pred[0]]}")
