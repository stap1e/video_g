import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np

class FVDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Precompute InceptionV3 features or use a pretrained model
        self.inception = self._load_inception_model()
    
    def _load_inception_model(self):
        # Simplified version - in practice use pretrained InceptionV3
        # For video, use 3D Inception or extract frame features
        class InceptionFeatureExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified feature extractor
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
            
            def forward(self, x):
                # x: (B, C, H, W)
                features = self.conv(x)
                return features.flatten(1)
        
        model = InceptionFeatureExtractor().to(self.device)
        model.eval()
        return model
    
    def compute_fvd(self, real_videos, fake_videos):
        # real_videos: (B, C, T, H, W)
        # fake_videos: (B, C, T, H, W)
        
        # Extract frame features
        real_features = []
        fake_features = []
        
        for t in range(real_videos.size(2)):
            real_frame = real_videos[:, :, t, :, :]
            fake_frame = fake_videos[:, :, t, :, :]
            
            with torch.no_grad():
                real_feat = self.inception(real_frame)
                fake_feat = self.inception(fake_frame)
            
            real_features.append(real_feat)
            fake_features.append(fake_feat)
        
        # Average over time
        real_feat = torch.cat(real_features, dim=0).mean(0)
        fake_feat = torch.cat(fake_features, dim=0).mean(0)
        
        # Compute FVD
        mu_real = real_feat.cpu().numpy()
        mu_fake = fake_feat.cpu().numpy()
        
        sigma_real = np.cov(real_feat.cpu().numpy().T)
        sigma_fake = np.cov(fake_feat.cpu().numpy().T)
        
        # Compute sqrt of product of covariances
        covmean = sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
        return fid

class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = self._load_inception_model()
    
    def _load_inception_model(self):
        # Simplified InceptionV3 feature extractor
        class InceptionFeatureExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
            
            def forward(self, x):
                features = self.conv(x)
                return features.flatten(1)
        
        model = InceptionFeatureExtractor().to(self.device)
        model.eval()
        return model
    
    def compute_fid(self, real_images, fake_images):
        # real_images: (B, C, H, W)
        # fake_images: (B, C, H, W)
        
        with torch.no_grad():
            real_features = self.inception(real_images)
            fake_features = self.inception(fake_images)
        
        mu_real = real_features.mean(dim=0).cpu().numpy()
        mu_fake = fake_features.mean(dim=0).cpu().numpy()
        
        sigma_real = np.cov(real_features.cpu().numpy().T)
        sigma_fake = np.cov(fake_features.cpu().numpy().T)
        
        covmean = sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
        return fid

# Simplified versions for quick use
def compute_fvd(real_videos, fake_videos, device='cuda'):
    calculator = FVDCalculator(device=device)
    return calculator.compute_fvd(real_videos, fake_videos)

def compute_fid(real_images, fake_images, device='cuda'):
    calculator = FIDCalculator(device=device)
    return calculator.compute_fid(real_images, fake_images)