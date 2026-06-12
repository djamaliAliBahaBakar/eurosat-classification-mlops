# EuroSAT Image Classification - MLOps Pipeline

## Overview

This project is a Machine Learning Engineering project for satellite image classification using the EuroSAT dataset.

The objective is not only to train a computer vision model, but to structure the project as a reproducible ML pipeline with clear separation between data loading, preprocessing, model building, training, evaluation, and inference.

The project uses MobileNetV3Small with transfer learning to classify satellite images into 10 land-use and land-cover classes.

## Problem

EuroSAT is a satellite image classification dataset containing 27,000 images across 10 classes.

The goal is to classify each image into one of the following categories:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV3Small
* scikit-learn
* NumPy
* Matplotlib
* YAML configuration
* Git / GitHub

## Project Structure

```text
eurosat-classification-mlops/
├── config/
│   ├── mobilenetv3_config.yaml
│   └── mobilenetv3_smoke_test.yaml
├── models/
│   ├── best_model_feature_extraction.keras
│   ├── best_model_finetuned.keras
│   └── class_names.json
├── scripts/
│   ├── smoke_test.py
│   └── predict_test.py
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── eurosat_classifier/
│       └── config.py
└── README.md
```

## ML Pipeline

The pipeline is organized into independent modules:

### 1. Data loading

`data.py` is responsible for loading the EuroSAT dataset and creating the training, validation, and test datasets.

### 2. Preprocessing

`preprocessing.py` applies image preprocessing and data augmentation.

The training dataset uses augmentation, while validation and test datasets only use the required preprocessing.

### 3. Model

`model.py` builds a MobileNetV3Small-based model.

The model uses ImageNet pretrained weights and adds a custom classification head for the 10 EuroSAT classes.

### 4. Training

`train.py` contains two training phases:

* Feature extraction: the MobileNetV3Small backbone is frozen.
* Fine-tuning: the last layers of the backbone are unfrozen and trained with a lower learning rate.

### 5. Evaluation

`evaluate.py` evaluates the trained model using:

* Loss
* Accuracy
* Top-2 accuracy
* Confusion matrix
* Classification report

### 6. Prediction

`predict.py` handles inference on a single image.

It loads and preprocesses an image, runs prediction, and returns:

* predicted class
* confidence score
* top-k predictions
* optional full probability distribution

## Configuration

Training parameters are defined in YAML files.

Example:

```yaml
training:
  epochs_frozen: 40
  epochs_finetune: 15
  learning_rate_frozen: 0.001
  learning_rate_finetune: 0.00001
```

This makes experiments easier to reproduce and avoids hardcoded training parameters.

## Smoke Test

A smoke test is provided to validate the full pipeline quickly.

It runs the complete workflow on a reduced number of batches:

```text
data loading
→ preprocessing
→ model creation
→ feature extraction
→ fine-tuning
→ evaluation
```

Run:

```bash
python3 scripts/smoke_test.py
```

The goal of the smoke test is not to achieve high accuracy, but to ensure that the full pipeline is correctly connected.

## Prediction Test

A prediction test script is provided to validate inference on a local image.

Run:

```bash
python3 scripts/predict_test.py
```

Example output:

```python
{
    "predicted_class": "Pasture",
    "confidence": 0.17,
    "topk": [
        ("Pasture", 0.17),
        ("Forest", 0.15),
        ("PermanentCrop", 0.13)
    ]
}
```

Low confidence is expected when using a smoke-test model trained on only a small subset of the dataset.

## Current Status

Implemented:

* Dataset loading
* Preprocessing pipeline
* MobileNetV3Small model
* Feature extraction training
* Fine-tuning
* Evaluation
* Smoke test
* Single-image prediction

Next steps:

* Train the full model with the complete configuration
* Add FastAPI inference endpoint
* Add Docker support
* Add GitHub Actions
* Improve experiment tracking

## Goal of the Project

This project demonstrates a practical ML Engineering workflow:

* modular codebase
* reproducible configuration
* transfer learning
* evaluation pipeline
* inference pipeline
* preparation for API deployment

The focus is on building a production-oriented ML project, not only achieving high model accuracy in a notebook.
