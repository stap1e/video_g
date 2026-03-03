import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


def _inception_model(device):
    if not hasattr(_inception_model, "model"):
        try:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=False)
        except Exception:
            model = inception_v3(weights=None, aux_logits=False)
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


def _predict_probs(images, device, batchsize):
    model = _inception_model(device)
    images = _preprocess(images)
    probs = []
    for start in range(0, images.size(0), batchsize):
        end = min(start + batchsize, images.size(0))
        with torch.no_grad():
            logits = model(images[start:end])
            prob = torch.softmax(logits, dim=1)
        probs.append(prob)
    return torch.cat(probs, dim=0)


def compute_inception_score_from_probs(probs, splits=10, eps=1e-20):
    probs = probs.cpu().numpy()
    n = probs.shape[0]
    scores = []
    for i in range(splits):
        part = probs[(i * n // splits):((i + 1) * n // splits), :]
        part = part + eps
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score_from_videos(videos, device="cuda", batchsize=32, n_frames=None, splits=10):
    frames = _select_frames(videos, n_frames)
    probs = _predict_probs(frames, device, batchsize)
    return compute_inception_score_from_probs(probs, splits=splits)


def compute_inception_score_from_sampler(sample_fn, n_samples=1000, batchsize=32, n_frames=None, splits=10, device="cuda"):
    probs_list = []
    collected = 0
    while collected < n_samples:
        n = min(batchsize, n_samples - collected)
        videos = sample_fn(n)
        frames = _select_frames(videos, n_frames)
        probs = _predict_probs(frames, device, batchsize)
        probs_list.append(probs)
        collected += n
    probs = torch.cat(probs_list, dim=0)
    return compute_inception_score_from_probs(probs, splits=splits)
