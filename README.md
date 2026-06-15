# Deepfake Audio Detection using Transfer Learning

## Overview

Artificial intelligence has significantly advanced speech synthesis capabilities, enabling the generation of highly realistic human-like voices. Although synthetic speech has valuable applications in accessibility, entertainment, and virtual assistants, it has also introduced challenges related to impersonation, misinformation, fraud, and security.

This project presents an automated Deepfake Audio Detection framework that distinguishes authentic human speech from AI-generated speech. By transforming audio recordings into Mel Spectrogram representations and leveraging a transfer learning approach with EfficientNetB0, the system learns subtle acoustic patterns that are difficult to identify manually.

The final model achieves high classification performance and is capable of detecting manipulated speech across previously unseen audio samples.

---

## Objectives

The primary goal of this project is to build a robust audio classification system that can:

* Identify genuine human speech recordings
* Detect AI-generated or synthesized speech
* Generalize effectively to new audio samples
* Maintain reliable performance across evaluation datasets

---

## Live Application

The model has been deployed as an interactive Streamlit application where users can upload audio files and receive instant predictions.

**Demo:**
https://deepfake-audio-detection-sgkw8zmfzcjycpmvnwjkga.streamlit.app/

---

## Dataset Information

### Source Dataset

The project utilizes the **Fake-or-Real (FoR) Dataset**, a benchmark dataset designed for speech authenticity classification.

### Dataset Categories

The dataset contains two classes:

* Real Human Speech
* Synthetic / AI-Generated Speech

### Data Volume

Approximately **20,000 audio recordings** were processed for training, validation, and testing.

### Data Partitioning

The dataset was divided into:

* Training Set
* Validation Set
* Test Set

This separation ensures unbiased evaluation and improved model generalization.

---

## Data Processing Pipeline

### Audio Standardization

Before model training, all recordings undergo a preprocessing pipeline that includes:

* Audio loading and decoding
* Conversion to a single audio channel
* Uniform sampling rate adjustment
* Signal normalization
* Noise reduction and amplitude stabilization

### Spectrogram Generation

Each audio signal is converted into a Mel Spectrogram representation, transforming temporal speech information into a visual frequency-domain format suitable for deep learning.

### Why Mel Spectrograms?

Mel Spectrograms are widely used in speech-related tasks because they:

* Preserve essential vocal characteristics
* Represent frequency patterns effectively
* Mimic aspects of human auditory perception
* Integrate naturally with CNN-based architectures

---

## Feature Engineering

The generated spectrograms are further processed through:

* Resizing to 128 × 128 dimensions
* Channel expansion from grayscale to RGB
* Dataset randomization and shuffling
* Stratified train-validation-test splitting

These steps help create consistent inputs for model training.

---

## Model Design

The classification model is built using **EfficientNetB0**, a highly efficient convolutional neural network pretrained on ImageNet.

### Model Workflow

Audio Recording

↓

Mel Spectrogram Conversion

↓

Image Preprocessing

↓

EfficientNetB0 Backbone

↓

Global Average Pooling

↓

Fully Connected Layer (128 Neurons)

↓

Dropout Regularization (40%)

↓

Binary Classification Output

---

## Training Strategy

The model was trained using transfer learning to accelerate convergence and improve performance.

### Configuration

| Hyperparameter          | Setting             |
| ----------------------- | ------------------- |
| Framework               | TensorFlow / Keras  |
| Backbone Network        | EfficientNetB0      |
| Optimizer               | Adam                |
| Loss Function           | Binary Crossentropy |
| Batch Size              | 64                  |
| Epochs                  | 15                  |
| Early Stopping          | Enabled             |
| Learning Rate Scheduler | Enabled             |

---

## System Workflow

1. Load raw audio recordings
2. Perform preprocessing and normalization
3. Generate Mel Spectrogram images
4. Resize spectrograms to model input size
5. Convert images into RGB format
6. Train EfficientNetB0 classifier
7. Validate model performance
8. Compute evaluation metrics
9. Deploy model through Streamlit interface

---

## Experimental Results

The proposed system demonstrates strong classification capability on the test dataset.

### Performance Metrics

| Metric                 | Result |
| ---------------------- | ------ |
| Accuracy               | 95.45% |
| F1 Score               | 95.50% |
| Equal Error Rate (EER) | 4.47%  |

These results indicate the model's effectiveness in distinguishing real speech from synthetic audio.

---

## Evaluation Summary

| Requirement         | Target | Outcome  |
| ------------------- | ------ | -------- |
| Accuracy            | ≥ 80%  | Achieved |
| F1 Score            | ≥ 80%  | Achieved |
| Equal Error Rate    | ≤ 12%  | Achieved |
| Class-wise Accuracy | ≥ 75%  | Achieved |

All evaluation criteria specified in the project requirements were successfully satisfied.

---

## Confusion Matrix

| Ground Truth    | Predicted Real | Predicted Fake |
| --------------- | -------------- | -------------- |
| Real Speech     | 1887           | 123            |
| Deepfake Speech | 59             | 1931           |

The confusion matrix demonstrates balanced performance across both classes with relatively few misclassifications.

---

## Technology Stack

* Python
* TensorFlow
* Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

---

## Project Directory

```text
Deepfake-Audio-Detection
│
├── notebook
│   └── Deepfake_Audio_Detection.ipynb
│
├── models
│   └── deepfake_audio_model_20k.keras
│
├── reports
│   ├── README.md
│   └── confusion_matrix.png
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Future Enhancements

Potential improvements include:

* Multi-class deepfake source identification
* Transformer-based audio classification models
* Real-time audio stream analysis
* Explainable AI visualizations
* Cross-dataset generalization studies
* Lightweight edge-device deployment

---

## Conclusion

This project demonstrates how transfer learning and spectrogram-based feature extraction can be effectively combined to build a high-performance Deepfake Audio Detection system. By leveraging EfficientNetB0 and carefully engineered preprocessing techniques, the solution achieves strong accuracy and reliability while remaining suitable for practical deployment through a user-friendly web application.
