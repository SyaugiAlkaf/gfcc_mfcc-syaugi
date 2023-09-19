# GFCC to RNN Prediction

This project demonstrates the process of extracting Gammatone Frequency Cepstral Coefficients (GFCC) from audio files and using them as input features for a Recurrent Neural Network (RNN) to predict patient conditions.

## Overview

1. **GFCC Extraction**: GFCCs are extracted from audio files. These coefficients are similar to MFCCs but are based on the Gammatone filter bank, which is designed to mimic the human auditory system.
2. **Data Preparation**: The extracted GFCCs are structured and labeled according to the patient's condition.
3. **RNN Model**: An RNN model, specifically using LSTM layers, is trained on the GFCC features to predict the patient's condition.

## Steps

### 1. GFCC Extraction

- **Gammatone Filter Bank**: A filter bank based on the Gammatone function is used. This mimics the frequency selectivity of the human auditory system.
- **GFCC Calculation**: The power spectrum of the audio signal is passed through the Gammatone filter bank. The resulting energies are then log-transformed and a Discrete Cosine Transform (DCT) is applied to obtain the GFCCs.

### 2. Data Preparation

- **Labeling**: Each audio file is labeled based on the patient's condition. This information is derived from a provided CSV file.
- **Feature Structuring**: The GFCCs are structured into a suitable format for input into the RNN.

### 3. RNN Model

- **Model Architecture**: The RNN model uses LSTM layers, which are a type of RNN layer suitable for sequence data like audio. Dropout layers are added for regularization to prevent overfitting.
- **Training**: The model is trained on a subset of the data (training set) and validated on another subset (validation/test set).
- **Prediction**: Once trained, the model can predict the condition of a patient based on the GFCCs extracted from their audio file.

## Usage

1. **Dependencies**:
    ```bash
    pip install numpy pandas sklearn tensorflow
    ```

2. **Run the GFCC extraction script** to obtain the GFCC features and labels from the audio files.

3. **Split the data** into training and test/validation sets.

4. **Train the RNN model** on the training data.

5. **Evaluate** the model's performance on the test/validation data.

6. **Predict**: Use the trained model to make predictions on new audio data.

## Conclusion

This project showcases the potential of using audio signal processing techniques combined with deep learning to predict medical conditions. The use of GFCCs provides a robust representation of the audio signal, and the RNN model can capture the sequential patterns in the data for accurate predictions.
