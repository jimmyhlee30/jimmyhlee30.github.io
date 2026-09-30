import cv2
import numpy as np
from scipy.signal import convolve2d

from filters import gaussian_kernel


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return cv2.sepFilter2D(image.astype(np.float32), -1, g, g, borderType=cv2.BORDER_REFLECT).astype(np.float64)


def per_channel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return convolve2d(image, kernel, mode="same", boundary="symm")
    return np.dstack([per_channel(image[..., c], kernel) for c in range(image.shape[2])])


def unsharp_kernel(sigma: float, alpha: float) -> np.ndarray:
    g = gaussian_kernel(sigma)
    impulse = np.zeros_like(g)
    impulse[g.shape[0] // 2, g.shape[1] // 2] = 1.0
    return (1 + alpha) * impulse - alpha * g


def sharpen(image: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    return np.clip(per_channel(image, unsharp_kernel(sigma, alpha)), 0, 1)


def low_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    return blur(image, sigma)


def high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    return image - blur(image, sigma)


def hybrid_image(im_high: np.ndarray, im_low: np.ndarray, sigma_high: float, sigma_low: float) -> np.ndarray:
    return np.clip(low_pass(im_low, sigma_low) + high_pass(im_high, sigma_high), 0, 1)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else image @ np.array([0.299, 0.587, 0.114])


def log_spectrum(image: np.ndarray) -> np.ndarray:
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(to_gray(image)))) + 1e-8)


def gaussian_stack(image: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    stack = [image.astype(np.float64)]
    for i in range(levels - 1):
        stack.append(blur(stack[-1], sigma * 2 ** i))
    return stack


def laplacian_stack(image: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    g = gaussian_stack(image, levels, sigma)
    return [g[i] - g[i + 1] for i in range(levels - 1)] + [g[-1]]


def blend(im_a: np.ndarray, im_b: np.ndarray, mask: np.ndarray, levels: int = 6, sigma: float = 2.0,
          mask_sigma: float | None = None):
    if im_a.ndim == 3 and mask.ndim == 2:
        mask = np.dstack([mask] * im_a.shape[2])
    la, lb = laplacian_stack(im_a, levels, sigma), laplacian_stack(im_b, levels, sigma)
    gm = gaussian_stack(mask.astype(np.float64), levels, mask_sigma or sigma)
    part_a = [m * l for m, l in zip(gm, la)]
    part_b = [(1 - m) * l for m, l in zip(gm, lb)]
    levels_out = [a + b for a, b in zip(part_a, part_b)]
    return np.clip(sum(levels_out), 0, 1), part_a, part_b, levels_out, gm
