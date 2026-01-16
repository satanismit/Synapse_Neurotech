# sEMG Gesture Classification Pipeline

This repository contains a complete machine learning pipeline for classifying 5 hand gestures using 8-channel surface EMG (sEMG) signals.

## Dataset
The dataset consists of sEMG recordings from 25 subjects across 3 sessions.
- **Input**: 8-channel sEMG signals (512 Hz).
- **Window**: 5 seconds (2560 samples).
- **Classes**: 5 gestures (0-4).

## Pipeline Overview

### 1. Preprocessing
- **Notch Filter**: 50Hz to remove power line interference.
- **Bandpass Filter**: 20-200Hz (4th order Butterworth) to isolate EMG frequencies and remove motion artifacts/high-freq noise.
- **Normalization**: Z-score standardization (zero mean, unit variance) per channel.

### 2. Model Architecture (`EMGConv1D`)
A lightweight 1D Convolutional Neural Network (CNN) designed for temporal signal classification:
- 4 Convolutional Blocks (Conv1D + BatchNorm + ReLU + MaxPool).
- Global Average Pooling to reduce parameters and prevent overfitting.
- Dropout (0.5) for regularization.
- Fully Connected Output Layer.

### 3. Training Strategy
- **Split**: Subject-independent split.
    - **Train**: Subjects 1-20.
    - **Validation**: Subjects 21-25.
- **Optimizer**: Adam (LR=0.001).
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5).
- **Early Stopping**: Patience of 10 epochs on Validation F1 Score.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model from scratch:
```bash
python train.py
```
This will save the best model to `output/best_model.pth`.

### Inference
To predict gestures on new CSV files, you have two options:

**Option 1: Interactive Demo (Recommended)**
Double-click `run_inference.bat` or run:
```bash
python Inference_pipeline.py
```
This will launch an interactive session where you can paste file paths or test on random dataset files.

**Option 2: Command Line**
```bash
python inference.py path/to/file.csv
# OR
python inference.py path/to/directory/
```

## File Structure
- `config.py`: Configuration parameters.
- `dataset.py`: Data loading and preprocessing logic.
- `model.py`: PyTorch model definition.
- `train.py`: Training loop and validation.
- `inference.py`: Core inference logic.
- `Inference_pipeline.py`: Interactive inference demo script.
- `run_inference.bat`: Windows batch file to run the demo.
- `requirements.txt`: Dependencies.
- `report.tex`: Technical report.

