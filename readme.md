# Video Diffusion Project

## Baseline Model
- **Model**: Google Video Diffusion Model
- **Backbone**: U-Net
- **Framework**: PyTorch

## Datasets

### Example
```
- **Dataset Name**: UCF101
  - **Description**: 101 action classes
  - **Size**: 13,320 videos
  - **Format**: AVI
  - **Source**: https://www.crcv.ucf.edu/data/UCF101.php
  - **License**: MIT
```

### Example
```
- **Dataset Name**: Kinetics 600
  - **Description**:
  - **Size**:  videos
  - **Format**: 
  - **Source**: https://opendatalab.com/OpenMMLab/Kinetics600/
  - **License**: 
```

### Example
```
- **Dataset Name**: BAIR_robot_pushing_dataset_v0
  - **Description**:
  - **Size**:  videos
  - **Format**: 
  - **Source**: http://www.google.com/url?q=http%3A%2F%2Frail.eecs.berkeley.edu%2Fdatasets%2Fbair_robot_pushing_dataset_v0.tar&sa=D&sntz=1&usg=AOvVaw3nXoeXKb5FbfJZIUaMX0p_
  - **License**: 
```

### Example
```
- **Dataset Name**: Kinetics 600
  - **Description**:
  - **Size**:  videos
  - **Format**: 
  - **Source**: https://opendatalab.com/OpenMMLab/Kinetics600/
  - **License**: 
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