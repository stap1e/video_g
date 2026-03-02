import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UCF101Dataset(Dataset):
    def __init__(self, data_path, image_size=64, time_steps=16, transform=None, split_file=None):
        self.data_path = data_path
        self.image_size = image_size
        self.time_steps = time_steps
        self.transform = transform
        
        # Get all class directories
        self.classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get video paths from split file if provided
        self.video_paths = []
        if split_file:
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        video_rel_path = line.split()[0]
                        self.video_paths.append(os.path.join(data_path, video_rel_path))
        else:
            # Fallback to loading all videos
            for cls in self.classes:
                cls_dir = os.path.join(data_path, cls)
                for video_file in os.listdir(cls_dir):
                    if video_file.endswith('.avi'):
                        self.video_paths.append(os.path.join(cls_dir, video_file))
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Read frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # If video has fewer frames than time_steps, repeat
        if len(frames) < self.time_steps:
            frames += [frames[-1]] * (self.time_steps - len(frames))
        
        # Select time_steps frames evenly
        indices = np.linspace(0, len(frames)-1, self.time_steps, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        # Apply transforms
        if self.transform:
            selected_frames = [self.transform(frame) for frame in selected_frames]
        
        # Stack frames into tensor (C, T, H, W)
        video_tensor = torch.stack(selected_frames).permute(1, 0, 2, 3)
        
        return video_tensor

def get_dataloader(data_path, batch_size=4, image_size=64, time_steps=16, shuffle=True, split_file=None):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = UCF101Dataset(data_path, image_size, time_steps, transform, split_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    return dataloader