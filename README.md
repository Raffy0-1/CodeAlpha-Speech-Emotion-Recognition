# Speech Emotion Recognition

A deep learning system that recognizes human emotions from speech audio. The model is trained on the RAVDESS and TESS datasets and classifies speech into eight emotion categories: angry, calm, disgust, fearful, happy, neutral, sad, and surprised.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This project applies speech signal processing and deep learning to the problem of automatic emotion recognition. The pipeline covers everything from raw audio ingestion to inference on new audio files, including data augmentation, feature extraction, model training, and evaluation.

The system was developed and trained on Kaggle using a free GPU (T4). The notebook is self-contained and can be run end to end on Kaggle by attaching the RAVDESS and TESS datasets from the Kaggle dataset library.

---

## Architecture

The model is a hybrid CNN + Bidirectional LSTM with an Attention mechanism.

```
Input (139, 200, 1)
        |
   CNN Block 1        Conv2D(64) x2, BatchNorm, ReLU, MaxPool, Dropout
        |
   CNN Block 2        Conv2D(128) x2, BatchNorm, ReLU, MaxPool, Dropout
        |
   Reshape            Merge frequency and channel dims, keep time axis
        |
   BiLSTM Layer 1     Bidirectional LSTM(128), return sequences
        |
   BiLSTM Layer 2     Bidirectional LSTM(64), return sequences
        |
   Attention          Dense(1, tanh) -> Softmax -> Weighted sum over time
        |
   Dense(256)         ReLU, BatchNorm, Dropout(0.4)
        |
   Dense(128)         ReLU, Dropout(0.2)
        |
   Dense(8)           Softmax output
```

The CNN layers extract local spectro-temporal patterns from the feature maps. The BiLSTM layers model temporal dependencies in both directions across the sequence. The attention mechanism learns to weight time steps by their emotional relevance so the model focuses on the most informative frames rather than treating all time steps equally.

**Loss function:** Focal Loss (gamma=2.0, alpha=0.25) with balanced class weights. Focal loss down-weights easy examples and focuses training on hard misclassifications, which helps with the class imbalance present in RAVDESS.

**Optimizer:** Adam with cosine annealing learning rate schedule and a 5-epoch linear warmup.

---

## Dataset

Two publicly available datasets are used:

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,440 speech audio files
- 24 professional actors (12 male, 12 female)
- 8 emotion classes
- Kaggle: `uwrfkaggler/ravdess-emotional-speech-audio`

**TESS** (Toronto Emotional Speech Set)
- 2,800 audio files
- Female speakers only
- 7 emotion classes (no calm category)
- Kaggle: `ejlok1/toronto-emotional-speech-set-tess`

Combined, the datasets provide approximately 4,240 base samples before augmentation.

**Train / Val / Test Split:**
The split is performed at the actor level to prevent actor identity leakage. If the same speaker appears in both train and test sets, the model can learn voice characteristics rather than emotions, producing inflated accuracy that does not generalize.

- Train: 70% of RAVDESS actors + all TESS samples
- Validation: 15% of RAVDESS actors
- Test: 15% of RAVDESS actors

After 9x augmentation on the training set, the total training size is approximately 55,000 samples.

---

## Feature Extraction

Each audio file is converted into a 2D feature matrix of shape (139, 200):

| Feature | Dimensions | Description |
|---|---|---|
| MFCCs | 40 | Mel-Frequency Cepstral Coefficients |
| MFCC Delta | 40 | First derivative of MFCCs (rate of change) |
| MFCC Delta-Delta | 40 | Second derivative of MFCCs (acceleration) |
| Spectral Contrast | 7 | Energy difference between spectral peaks and valleys |
| Chroma | 12 | Pitch class profiles |
| Total | 139 | Per time frame, 200 time frames |

**Preprocessing steps applied to every file:**
1. Pre-emphasis filter to boost high-frequency content
2. Silence trimming from edges (top_db=20)
3. Per-feature normalization (zero mean, unit variance per row)
4. Padding or truncation to fixed length of 200 time frames

**Data augmentation applied to training set only (9x expansion):**
- Pitch shift up 3 semitones
- Pitch shift down 3 semitones
- Time stretch slow (rate=0.85)
- Time stretch fast (rate=1.15)
- White noise injection
- Volume scale up (1.2x)
- Volume scale down (0.8x)
- Pitch shift + noise (compound)

SpecAugment (random time and frequency masking) is applied to all augmented versions but not the original.

---

## Project Structure

```
speech-emotion-recognition/
|
|-- speech-emotion-recognition-kaggle.ipynb    Main notebook
|
|-- results/
|   |-- training_history.png                   Accuracy and loss curves
|   |-- confusion_matrix.png                   Per-class confusion matrix
|   |-- per_class_metrics.png                  Precision, recall, F1 per emotion
|   |-- data_distribution.png                  Emotion and dataset distribution
|   |-- example_prediction.png                 Sample inference visualization
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- LICENSE
```

The model weights, scaler, label encoder, and metadata are saved during training but are not included in the repository due to file size. They are generated when you run the notebook.

---

## Setup and Usage

### Running on Kaggle (Recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and create a new notebook
2. Click Add Data in the right panel and attach:
   - `ravdess-emotional-speech-audio` by uwrfkaggler
   - `toronto-emotional-speech-set-tess` by ejlok1
3. Go to Settings and set Accelerator to GPU T4
4. Upload `speech-emotion-recognition-kaggle.ipynb` and run all cells in order

### Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Update the path variables in Cell 2 to point to your local directories:

```python
PROCESSED_DIR = Path('./processed')
MODEL_DIR     = Path('./models')
RESULTS_DIR   = Path('./results')
RAVDESS_DIR   = Path('/path/to/ravdess')
TESS_DIR      = Path('/path/to/tess')
```

Then run all cells in order. Feature extraction for the full training set with 9x augmentation takes approximately 2.5 hours on CPU.

### Running Inference on a New Audio File

After training, use the `EmotionRecognizer` class from Cell 17:

```python
recognizer = EmotionRecognizer(
    model_path    = 'path/to/best_model.keras',
    metadata_path = 'path/to/model_metadata.json'
)

result = recognizer.predict('path/to/audio.wav')
print(result['emotion'])
print(result['confidence'])

# For a visualization with waveform, spectrogram, and probability bars
recognizer.visualize('path/to/audio.wav', true_emotion='happy')
```

The audio file should be a WAV file, ideally 3-4 seconds long. The recognizer handles loading, preprocessing, and normalization internally using the same scaler fitted during training.

---

## Results

Results are stored in the `results/` folder.

**Test set performance (RAVDESS held-out actors):**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| Angry | 0.571 | 0.500 | 0.533 |
| Calm | 0.655 | 0.594 | 0.623 |
| Disgust | 0.688 | 0.688 | 0.688 |
| Fearful | 0.609 | 0.438 | 0.509 |
| Happy | 0.429 | 0.656 | 0.519 |
| Neutral | 0.308 | 0.250 | 0.276 |
| Sad | 0.324 | 0.375 | 0.348 |
| Surprised | 0.724 | 0.656 | 0.689 |

Overall test accuracy: 53.8% on 240 held-out samples.

**Per-sample inference results (one sample per class):**

| True Emotion | Predicted | Confidence | Correct |
|---|---|---|---|
| Angry | Angry | 51.2% | Yes |
| Calm | Calm | 97.6% | Yes |
| Disgust | Disgust | 98.6% | Yes |
| Fearful | Fearful | 94.0% | Yes |
| Happy | Happy | 96.3% | Yes |
| Neutral | Neutral | 97.9% | Yes |
| Sad | Calm | 87.4% | No |
| Surprised | Surprised | 99.4% | Yes |

The model achieves 7 out of 8 correct on per-sample inference with high confidence. The sad/calm confusion is expected — both emotions share acoustic characteristics including low energy, slower speech rate, and similar fundamental frequency ranges. This confusion is documented in the speech emotion recognition literature and is not unique to this model.

The gap between overall test accuracy (53.8%) and per-sample inference quality is explained by overfitting visible in the training curves: training accuracy reached 99% while validation accuracy plateaued around 44%, indicating the model memorized training samples rather than learning fully generalizable representations. This is a known challenge when training on limited actor-level data with high augmentation ratios.

---

## Known Limitations

- The model was trained on acted speech from professional actors, not naturalistic speech. Real-world performance on spontaneous or conversational speech will be lower.
- The validation and test sets contain only 200 and 240 samples respectively due to the strict actor-level split on 24 actors. This makes accuracy metrics sensitive to individual predictions.
- The sad and calm emotions are frequently confused due to their acoustic similarity. A larger and more diverse dataset would help the model learn finer distinctions.
- The model is optimized for English speech. Performance on other languages has not been evaluated.

---

## License

MIT License. See LICENSE file for details.
