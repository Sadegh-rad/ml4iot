# Homework 1 - Audio Processing and Voice Activity Detection

## Description
This homework focuses on audio processing and voice activity detection using TensorFlow and audio processing libraries.

## Files
- `ex1.py` - Main audio processing and voice activity detection implementation
- `ex2.py` - Additional audio processing exercises
- `ML4IoT-HW1_Problem_description.pdf` - Problem statement and requirements
- `Team18_Homework1.pdf` - Final report and solutions

## Key Features
- Real-time audio recording and processing
- Voice activity detection with silence detection
- Audio file size optimization
- TensorFlow integration for audio processing

## Dependencies
- sounddevice
- scipy
- tensorflow
- tensorflow_io

## Usage
Run the main script:
```bash
python ex1.py
```

## Implementation Details
- Real-time audio capture using sounddevice
- Silence detection based on dBFS threshold
- Automatic audio file saving when voice activity is detected
- Configurable parameters for downsampling and frame length
