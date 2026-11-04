import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_gaussian_window(window_size: int, channel: int, device, dtype, sigma: float = 1.5):
	coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
	g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
	g = g / g.sum()
	kernel_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
	return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(
	img1: torch.Tensor,
	img2: torch.Tensor,
	window_size: int = 11,
	size_average: bool = True,
	data_range: float = 1.0,
	eps: float = 1e-12,
):
	"""Compute SSIM for NCHW tensors.

	Args:
		img1: Tensor in shape [N, C, H, W].
		img2: Tensor in shape [N, C, H, W].
		window_size: Gaussian kernel size used for local statistics.
		size_average: If True return scalar mean; else return per-sample SSIM.
		data_range: Dynamic range of image values (e.g., 1.0 for [0, 1]).
	"""
	if img1.shape != img2.shape:
		raise ValueError(f"img1/img2 shape mismatch: {img1.shape} vs {img2.shape}")
	if img1.dim() != 4:
		raise ValueError(f"SSIM expects NCHW tensors, got dim={img1.dim()}")

	img1 = img1.float()
	img2 = img2.float()

	_, c, h, w = img1.shape
	ws = min(window_size, h, w)
	ws = max(ws, 3)
	if ws % 2 == 0:
		ws -= 1

	window = _build_gaussian_window(ws, c, img1.device, img1.dtype)
	padding = ws // 2

	mu1 = F.conv2d(img1, window, padding=padding, groups=c)
	mu2 = F.conv2d(img2, window, padding=padding, groups=c)

	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=c) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=c) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=c) - mu1_mu2

	c1 = (0.01 * data_range) ** 2
	c2 = (0.03 * data_range) ** 2

	num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
	den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
	ssim_map = num / (den + eps)

	if size_average:
		return ssim_map.mean()
	return ssim_map.mean(dim=(1, 2, 3))


class SSIM(nn.Module):
	"""Module wrapper so it can be called as SSIM()(img1, img2)."""

	def __init__(self, window_size: int = 11, size_average: bool = True, data_range: float = 1.0):
		super().__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.data_range = data_range

	def forward(self, img1: torch.Tensor, img2: torch.Tensor):
		return ssim(
			img1,
			img2,
			window_size=self.window_size,
			size_average=self.size_average,
			data_range=self.data_range,
		)
