import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights


def _inception_model(device):
    if not hasattr(_inception_model, "model"):
        try:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=False)
        except Exception:
            model = inception_v3(weights=None, aux_logits=False)
        model.fc = torch.nn.Identity()
        model.to(device)
        model.eval()
        _inception_model.model = model
    return _inception_model.model


def _preprocess(images):
    images = (images + 1) / 2
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _select_frames(videos, n_frames):
    if videos.dim() == 4:
        return videos
    t = videos.size(2)
    if n_frames is None or n_frames >= t:
        indices = torch.arange(t, device=videos.device)
    else:
        indices = torch.linspace(0, t - 1, n_frames, device=videos.device).long()
    frames = videos[:, :, indices, :, :]
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()
    return frames.view(-1, videos.size(1), videos.size(3), videos.size(4))


def extract_features_from_videos(videos, device="cuda", batchsize=32, n_frames=None):
    model = _inception_model(device)
    frames = _select_frames(videos, n_frames)
    frames = _preprocess(frames)
    features = []
    for start in range(0, frames.size(0), batchsize):
        end = min(start + batchsize, frames.size(0))
        with torch.no_grad():
            feat = model(frames[start:end])
        features.append(feat)
    return torch.cat(features, dim=0)


def compute_fid_from_features(fake_features, real_features=None, stat_file=None):
    if stat_file and os.path.exists(stat_file):
        stat = np.load(stat_file)
        mu_real = stat["mean"]
        sigma_real = stat["cov"]
    else:
        mu_real = real_features.mean(dim=0).cpu().numpy()
        sigma_real = np.cov(real_features.cpu().numpy(), rowvar=False)
    mu_fake = fake_features.mean(dim=0).cpu().numpy()
    sigma_fake = np.cov(fake_features.cpu().numpy(), rowvar=False)
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_fake.ndim == 0:
        sigma_fake = np.array([[sigma_fake]])
    covmean = sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(np.real(fid))
