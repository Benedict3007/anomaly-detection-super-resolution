from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - allow import without torch
    torch = None  # type: ignore
    F = None  # type: ignore


def psnr_numpy(img_ref: np.ndarray, img: np.ndarray, data_range: Optional[float] = None) -> float:
    ref = img_ref.astype(np.float32)
    out = img.astype(np.float32)
    if data_range is None:
        data_range = 1.0 if np.issubdtype(ref.dtype, np.floating) else 255.0
    mse = float(np.mean((ref - out) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def ssim_numpy(img_ref: np.ndarray, img: np.ndarray, win_size: int = 11, data_range: Optional[float] = None) -> float:
    # Lightweight SSIM approximation using uniform kernel; expects HxW or HxWxC arrays in [0,1] or [0,255]
    ref = img_ref.astype(np.float32)
    out = img.astype(np.float32)
    if data_range is None:
        data_range = 1.0 if np.issubdtype(ref.dtype, np.floating) else 255.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # Convert to grayscale-like if 3 channels
    if ref.ndim == 3:
        if ref.shape[2] > 1:
            coeffs = np.array([65.738, 129.057, 25.064], dtype=np.float32) / 256.0
            ref = np.tensordot(ref, coeffs, axes=([2], [0]))
            out = np.tensordot(out, coeffs, axes=([2], [0]))
        else:
            # Squeeze singleton channel dimension to get HxW
            ref = ref[:, :, 0]
            out = out[:, :, 0]
    # Uniform filter via convolution using same-sized padding
    pad = win_size // 2
    kernel = np.ones((win_size, win_size), dtype=np.float32) / float(win_size * win_size)
    # naive convolution using scipy is avoided; implement via np.pad and strides
    def conv2(x: np.ndarray) -> np.ndarray:
        xpad = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
        h, w = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                region = xpad[i:i+win_size, j:j+win_size]
                out[i, j] = float(np.sum(region * kernel))
        return out

    mu1 = conv2(ref)
    mu2 = conv2(out)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv2(ref * ref) - mu1_sq
    sigma2_sq = conv2(out * out) - mu2_sq
    sigma12 = conv2(ref * out) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def psnr_torch(sr: "torch.Tensor", hr: "torch.Tensor", rgb_range: float) -> float:
    assert torch is not None and F is not None
    diff = (sr - hr).div(rgb_range)
    shave = 4  # match previous behavior
    if sr.size(-1) > 2 * shave:
        diff = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * float(np.log10(1.0 / mse))


def ssim_torch(sr: "torch.Tensor", hr: "torch.Tensor", rgb_range: float, win_size: int = 11) -> float:
    assert torch is not None and F is not None
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        sr = sr[..., :hr.size(-2), :hr.size(-1)]
    sr = sr.div(rgb_range).clamp(0, 1)
    hr = hr.div(rgb_range).clamp(0, 1)
    shave = 4
    if sr.size(-1) > 2 * shave:
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]
    if sr.size(1) > 1:
        convert = torch.tensor([[65.738, 129.057, 25.064]], dtype=sr.dtype, device=sr.device).view(1, 3, 1, 1) / 256
        sr = (sr * convert).sum(dim=1, keepdim=True)
        hr = (hr * convert).sum(dim=1, keepdim=True)
    C1 = (0.01 * 1.0) ** 2 * (255.0 ** 2)
    C2 = (0.03 * 1.0) ** 2 * (255.0 ** 2)
    kernel = torch.ones(1, 1, win_size, win_size, dtype=sr.dtype, device=sr.device) / float(win_size * win_size)
    mu1 = F.conv2d(sr, kernel, padding=win_size // 2)
    mu2 = F.conv2d(hr, kernel, padding=win_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(sr ** 2, kernel, padding=win_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(hr ** 2, kernel, padding=win_size // 2) - mu2_sq
    sigma12 = F.conv2d(sr * hr, kernel, padding=win_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


