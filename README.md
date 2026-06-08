# EuroSAT Classification MLOps

## Project Overview

This project demonstrates the design and implementation of a production-oriented Deep Learning pipeline for satellite image classification using the EuroSAT dataset.

The objective is to compare a baseline Convolutional Neural Network (CNN) trained from scratch with a transfer learning approach based on a pre-trained MobileNetV3 model.

Several training strategies are explored, including feature extraction and fine-tuning, in order to measure the impact of transfer learning on model performance.

The project follows an end-to-end Machine Learning workflow covering:

- Data preparation
- Data augmentation
- Model training
- Transfer learning
- Fine-tuning
- Model evaluation
- Experiment comparison
- Reproducibility
- Deployment readiness

The final goal is to demonstrate the ability to build production-oriented Deep Learning systems rather than focusing solely on model accuracy.

---

## Problem Statement

Land-cover classification is a common Computer Vision task in Earth Observation.

Given a satellite image, the objective is to predict the corresponding land-cover category such as:

- Forest
- River
- Residential
- Highway
- Pasture
- Industrial
- Annual Crop
- Permanent Crop
- Herbaceous Vegetation
- Sea/Lake

Accurate classification of satellite imagery is useful for environmental monitoring, urban planning and agricultural analysis.

---

## Dataset

This project uses the EuroSAT dataset.

EuroSAT is a publicly available benchmark dataset composed of labeled Sentinel-2 satellite images covering 10 land-use and land-cover classes.

Dataset characteristics:

- 10 classes
- RGB satellite imagery
- Balanced dataset
- ~27,000 labeled images

---

## Methodology

### Baseline Model

A custom CNN is trained from scratch to establish a baseline performance.

### Transfer Learning

A pre-trained MobileNetV3 model is used as a feature extractor.

### Fine-Tuning

Selected layers of the MobileNetV3 backbone are unfrozen and fine-tuned on the EuroSAT dataset.

The objective is to evaluate how transfer learning improves classification performance compared to training a model from scratch.

---

## Experiments

The following experiments are performed:

| Experiment | Model | Strategy |
|------------|--------|----------|
| Baseline | Custom CNN | Training from scratch |
| Experiment 1 | MobileNetV3 | Feature Extraction |
| Experiment 2 | MobileNetV3 | Fine-Tuning |

---

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics provide a more complete view of model performance than accuracy alone.

---

## Results

Results will be added after the completion of the training pipeline.

| Experiment | Accuracy | Precision | Recall | F1 |
|------------|------------|------------|------------|------------|
| Baseline CNN | TBD | TBD | TBD | TBD |
| MobileNetV3 Feature Extraction | TBD | TBD | TBD | TBD |
| MobileNetV3 Fine-Tuning | TBD | TBD | TBD | TBD |

---

## Inference Example

Example of the expected inference workflow:

Input image:

```text
satellite_image.jpg
```

Prediction:

```text
Forest        94%
River          4%
Pasture        2%
```

The final system should be able to process unseen satellite images and provide class probabilities for decision support.

---

## Project Structure

```text
eurosat-classification-mlops/
│
├── notebooks/
├── src/
├── configs/
├── models/
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

---

## Reproducibility

The project aims to follow Machine Learning Engineering best practices:

- Modular codebase
- Configuration-driven training
- Experiment tracking
- Reproducible training pipeline
- Deployment-ready artifacts

---

## Future Improvements

Potential future extensions:

- Compare additional architectures (EfficientNet, ResNet)
- Hyperparameter tuning
- MLflow integration
- Docker packaging
- CI/CD pipeline
- Model serving API