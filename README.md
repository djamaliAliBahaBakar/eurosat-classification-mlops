# EuroSAT Image Classification - MLOps Pipeline

## Overview

This project is a Machine Learning Engineering project for satellite image classification using the EuroSAT dataset.

The objective is not only to train a computer vision model, but also to build a reproducible and production-oriented ML pipeline with clear separation between data loading, preprocessing, model training, evaluation, inference, API serving, and containerization.

The project uses MobileNetV3Small with transfer learning to classify satellite images into 10 land-use and land-cover classes.

---

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

---

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV3Small
* FastAPI
* Docker
* scikit-learn
* NumPy
* Matplotlib
* YAML configuration
* Git / GitHub

---

## Architecture

```text
EuroSAT Dataset
        ↓
Data Loading
        ↓
Preprocessing & Augmentation
        ↓
MobileNetV3Small
        ↓
Feature Extraction
        ↓
Fine-Tuning
        ↓
Model Artifact (.keras)
        ↓
Inference Pipeline
        ↓
FastAPI REST API
        ↓
Docker Container
```

---

## Model Performance

Results will be updated after full training on the complete dataset configuration.

Current smoke-test results are intended only to validate the end-to-end pipeline and should not be used to evaluate model quality.

---

## Project Structure

```text
eurosat-classification-mlops/
├── api/
│   └── main.py
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
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ML Pipeline

The pipeline is organized into independent modules.

### 1. Data Loading

`data.py` is responsible for loading the EuroSAT dataset and creating the training, validation, and test datasets.

### 2. Preprocessing

`preprocessing.py` applies image preprocessing and data augmentation.

The training dataset uses augmentation, while validation and test datasets only use the required preprocessing.

### 3. Model

`model.py` builds a MobileNetV3Small-based classifier.

The model uses ImageNet pretrained weights and adds a custom classification head for the 10 EuroSAT classes.

### 4. Training

`train.py` contains two training phases:

#### Feature Extraction

The MobileNetV3Small backbone is frozen and only the classification head is trained.

#### Fine-Tuning

The last layers of the backbone are unfrozen and trained using a smaller learning rate.

### 5. Evaluation

`evaluate.py` evaluates the trained model using:

* Loss
* Accuracy
* Top-2 Accuracy
* Confusion Matrix
* Classification Report

### 6. Prediction

`predict.py` handles inference on a single image.

It:

* Loads an image
* Applies preprocessing
* Runs model inference
* Returns:

  * Predicted class
  * Confidence score
  * Top-k predictions

---

## Configuration

Training parameters are defined in YAML configuration files.

Example:

```yaml
training:
  epochs_frozen: 40
  epochs_finetune: 15
  learning_rate_frozen: 0.001
  learning_rate_finetune: 0.00001
```

This approach improves reproducibility and avoids hardcoded parameters.

---

## Smoke Test

A smoke test validates the complete ML pipeline on a reduced subset of the dataset.

Workflow:

```text
Data Loading
→ Preprocessing
→ Model Creation
→ Feature Extraction
→ Fine-Tuning
→ Evaluation
```

Run:

```bash
python3 scripts/smoke_test.py
```

The objective is pipeline validation, not model performance.

---

## Local Prediction Test

A prediction script validates inference on a local image.

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

Low confidence is expected when using a smoke-test model.

---

## FastAPI Inference API

The project exposes a REST API using FastAPI.

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Prediction Endpoint

```http
POST /predict
```

Upload an image and receive model predictions.

Example:

```bash
curl -X POST \
  -F "file=@data/Sample-images.ppm" \
  http://127.0.0.1:8000/predict
```

Example response:

```json
{
  "predicted_class": "Pasture",
  "confidence": 0.175,
  "topk": [
    ["Pasture", 0.175],
    ["Forest", 0.158],
    ["PermanentCrop", 0.138]
  ]
}
```

---

## Docker

Build the Docker image:

```bash
docker build -t eurosat-api .
```

Run the container:

```bash
docker run -p 8000:8000 eurosat-api
```

Verify the API:

```bash
curl http://127.0.0.1:8000/health
```

Run a prediction:

```bash
curl -X POST \
  -F "file=@data/Sample-images.ppm" \
  http://127.0.0.1:8000/predict
```

---

## Current Status

Implemented:

* Dataset loading
* Preprocessing pipeline
* MobileNetV3Small model
* Feature extraction training
* Fine-tuning
* Evaluation pipeline
* Smoke test
* Single-image prediction
* FastAPI inference API
* Docker containerization

---

## Future Improvements

* Train the full model using the complete dataset
* Add GitHub Actions CI pipeline
* Add automated API tests
* Deploy the Dockerized API
* Add experiment tracking (MLflow)
* Add model monitoring
* Explore alternative CNN architectures

---

## Goal of the Project

This project demonstrates a complete Machine Learning Engineering workflow:

* Modular codebase
* Reproducible configuration
* Transfer learning
* Evaluation pipeline
* Inference pipeline
* REST API serving
* Dockerized deployment

The focus is on building a production-oriented ML project rather than only achieving high accuracy in a notebook environment.
