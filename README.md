# Audio Classification Project

## Overview

This project aims to classify audio files into different animal categories using a Convolutional Neural Network (CNN) model. The project is structured to leverage **ZenML** for managing steps and pipelines, **MLFlow** for experiment tracking, and **Streamlit** for building and deploying the application.

## Key Technologies

### ZenML
ZenML is used for orchestrating the different steps involved in the audio classification pipeline. The steps include:

- **Data Ingestion**: Loading audio files, extracting features (such as Mel spectrograms), and preparing the data for model training.
- **Data Splitting**: Splitting Data into Train and Test set.
- **Model Training**: Training a CNN model to classify audio files into predefined categories.
- **Model Evaluation**: Evaluating the model's performance using appropriate metrics.
- **Model Deployment**: To verify if the new trained model meets required criteria and then save the Trained model.

### MLFlow
MLFlow is integrated to track and log various aspects of the model training process. This includes:

- **Experiment Tracking**: Logging model parameters, metrics, and artifacts during training.
- **Model Versioning**: Keeping track of different versions of the model to ensure reproducibility.
- **Auto Logging**: Automatically capturing relevant data during the training process.

### Streamlit
Streamlit is utilized to create an interactive web application for real-time audio classification. The app features:

- **File Upload**: Users can upload `.wav` files, and the app will predict the corresponding animal category.
- **Audio Recording**: Users can record audio directly within the app, which is then processed and classified.
- **Real-Time Predictions**: The app provides immediate feedback on the predicted category of the uploaded or recorded audio.

## Project Workflow

1. **Data Ingestion**:
    - **Data Collection**: The audio data is stored in a predefined directory structure, where each subdirectory represents a different animal category (e.g., birds, cats, dogs). The ZenML pipeline begins by scanning this directory to identify all available audio files.
    - **Feature Extraction**: For each audio file, features are extracted to transform the raw audio into a form suitable for machine learning. This project uses Mel spectrograms, which are a representation of the power spectrum of the audio signals. Mel spectrograms are extracted using the `librosa` library, which provides robust tools for audio analysis.
    - **Data Preparation**: The extracted Mel spectrograms are then normalized and padded or trimmed to ensure uniform shape across all samples. This step is crucial for batch processing during model training. Finally, the data is split into features (`X`) and labels (`y`), where `X` contains the spectrogram data and `y` contains the corresponding labels representing different animal categories.

2. **Model Training**:
    - **Model Architecture**: The model used in this project is a Convolutional Neural Network (CNN) designed to handle 2D input data, like images or spectrograms. The architecture includes multiple convolutional layers to capture spatial features from the spectrograms, followed by max-pooling layers to reduce dimensionality. Dense layers are then used to map these features to output predictions, with a softmax layer at the end to generate probabilities for each class.
    - **Training Process**: The model is trained using the processed spectrograms as input. The training involves iterating over the dataset, adjusting the model's weights using backpropagation and an optimizer (such as Adam). The training process is monitored using various metrics, such as accuracy and loss, to ensure the model is learning effectively.
    - **MLFlow Integration**: During training, MLFlow is used to track various aspects of the experiment. This includes logging model parameters (e.g., learning rate, batch size), performance metrics (e.g., accuracy, loss), and artifacts (e.g., trained models). MLFlow's autologging feature ensures that all relevant information is captured without requiring manual logging.

3. **Model Evaluation**:
    - **Evaluation Metrics**: After training, the model is evaluated on a validation set to determine its performance. Common metrics include accuracy, precision, recall, and F1 score. These metrics provide insights into how well the model is likely to perform on unseen data.
    - **Performance Logging**: The evaluation metrics are logged into MLFlow, allowing for easy comparison with previous models and experiments. This helps in identifying the best-performing model and understanding the impact of any changes made to the model or data preprocessing steps.
    - **Model Saving**: Once evaluated, the best-performing model is saved for deployment. The model can be versioned and stored within MLFlow, ensuring that the exact model used for deployment is easily accessible and reproducible.

4. **Deployment**:
    - **Streamlit Application**: The trained model is deployed using a Streamlit application, providing an interactive interface for users. The app is designed with two main functionalities:
        - **File Upload**: Users can upload `.wav` files, which are then processed by the model to predict the corresponding animal category. The Mel spectrogram is generated from the uploaded audio, and the model's prediction is displayed in the app.
        - **Audio Recording**: The app also allows users to record audio directly from their microphone. This audio is immediately processed and classified by the model, providing a real-time prediction of the animal sound.
    - **Real-Time Predictions**: The app delivers real-time feedback, making it suitable for interactive use cases, such as education or entertainment. The user-friendly interface ensures that even those with no technical background can easily use the application.
    - **Production Deployment**: Streamlit simplifies the deployment process, making it possible to host the application on various platforms, including local servers or cloud services. The app can be accessed via a web browser, making it accessible to a broad audience.
