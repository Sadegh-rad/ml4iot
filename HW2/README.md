# Homework 2 - TensorFlow Lite Model Training and Deployment

## Description
This homework involves training machine learning models, converting them to TensorFlow Lite format, and implementing preprocessing pipelines for IoT deployment.

## Files
- `ex1.py` - Main implementation with Redis integration and model inference
- `preprocessing.py` - Data preprocessing utilities
- `training.ipynb` - Jupyter notebook for model training
- `model18.tflite` - Trained TensorFlow Lite model
- `ML4IoT-HW2_Problem_description.pdf` - Problem statement and requirements
- `Team18_Homework2.pdf` - Final report and solutions

## Key Features
- TensorFlow Lite model training and conversion
- Redis database integration for data storage
- Real-time audio processing and inference
- Resource monitoring (CPU, memory usage)
- Model optimization for IoT devices

## Dependencies
- numpy
- tensorflow
- redis
- psutil
- sounddevice

## Usage
Run the main script with Redis configuration:
```bash
python ex1.py --host <redis_host> --port <redis_port> --user <username> --password <password>
```

## Implementation Details
- Real-time audio capture and preprocessing
- TensorFlow Lite model inference
- Redis integration for data persistence
- System resource monitoring
- Optimized for edge device deployment
