# Video Diffusion Project

## Baseline Model
- **Model**: Google Video Diffusion Model
- **Backbone**: U-Net
- **Framework**: PyTorch

## Datasets

### Template
```
- **Dataset Name**: 
  - **Description**: 
  - **Size**: 
  - **Format**: 
  - **Source**: 
  - **License**: 
```

### Example
```
- **Dataset Name**: UCF101
  - **Description**: 101 action classes
  - **Size**: 13,320 videos
  - **Format**: AVI
  - **Source**: https://www.crcv.ucf.edu/data/UCF101.php
  - **License**: MIT
```

## Training Configuration
- **Batch Size**: 1
- **Epochs**: 100
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-6
- **Timesteps**: 1000
- **Objective**: pred_x0

## Evaluation Metrics
- **FVD (Fréchet Video Distance)**
- **FID (Fréchet Inception Distance)**

## Usage
```bash
# Train model
python main.py

# Evaluate model
python evaluate.py
```