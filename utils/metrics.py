import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np
from torchvision.models import vgg16, VGG16_Weights

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
        
        real_feat = torch.cat(real_features, dim=0)
        fake_feat = torch.cat(fake_features, dim=0)
        
        mu_real = real_feat.mean(dim=0).cpu().numpy()
        mu_fake = fake_feat.mean(dim=0).cpu().numpy()
        
        sigma_real = np.cov(real_feat.cpu().numpy(), rowvar=False)
        sigma_fake = np.cov(fake_feat.cpu().numpy(), rowvar=False)
        if sigma_real.ndim == 0:
            sigma_real = np.array([[sigma_real]])
        if sigma_fake.ndim == 0:
            sigma_fake = np.array([[sigma_fake]])
        
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

def _lpips_model(device):
    if not hasattr(_lpips_model, "model"):
        try:
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights).features[:16].to(device)
        except Exception:
            model = vgg16(weights=None).features[:16].to(device)
        model.eval()
        _lpips_model.model = model
    return _lpips_model.model

def compute_lpips(real_images, fake_images, device='cuda'):
    model = _lpips_model(device)
    real = (real_images + 1) / 2
    fake = (fake_images + 1) / 2
    mean = torch.tensor([0.485, 0.456, 0.406], device=real.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=real.device).view(1, 3, 1, 1)
    real = (real - mean) / std
    fake = (fake - mean) / std
    with torch.no_grad():
        real_feat = model(real)
        fake_feat = model(fake)
    return torch.mean((real_feat - fake_feat) ** 2).item()

def compute_psnr(real_images, fake_images):
    real = (real_images + 1) / 2
    fake = (fake_images + 1) / 2
    mse = torch.mean((real - fake) ** 2)
    if mse.item() == 0:
        return float("inf")
    value = 20 * torch.log10(torch.tensor(1.0, device=real.device)) - 10 * torch.log10(mse)
    return value.item()

def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window

def compute_ssim(real_images, fake_images, window_size=11, sigma=1.5):
    real = (real_images + 1) / 2
    fake = (fake_images + 1) / 2
    c = real.size(1)
    window = _gaussian_window(window_size, sigma, real.device, real.dtype)
    window = window.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)
    padding = window_size // 2
    mu1 = F.conv2d(real, window, padding=padding, groups=c)
    mu2 = F.conv2d(fake, window, padding=padding, groups=c)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(real * real, window, padding=padding, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(fake * fake, window, padding=padding, groups=c) - mu2_sq
    sigma12 = F.conv2d(real * fake, window, padding=padding, groups=c) - mu1_mu2
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()
