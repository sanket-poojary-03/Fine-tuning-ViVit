# Fine-Tuning ViVit with HuggingFace Trainer

This repository contains a Python script to fine-tune the open-source Video Vision Transformer (ViVit) model using the HuggingFace Trainer Library. The model has been configured for 10 classes.


## Introduction

The Video Vision Transformer (ViVit) is a state-of-the-art model for video understanding tasks. This repository provides a script to fine-tune the ViVit model on your custom dataset using the HuggingFace Trainer Library. The model is pre-configured to classify videos into 10 different classes.


## Installation

To get started, clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Preparation

Prepare your dataset in the following format:
```python
DatasetDict({
    train: Dataset({
        features: ['labels', 'pixel_values'],
        num_rows: 36
    })
    test: Dataset({
        features: ['labels', 'pixel_values'],
        num_rows: 4
    })
})
```

