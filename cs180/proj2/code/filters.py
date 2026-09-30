import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.signal import convolve2d

DX = np.array([[1.0, -1.0]])
DY = np.array([[1.0], [-1.0]])
BOX9 = np.ones((9, 9)) / 81.0


def _pad_for(image: np.ndarray, kernel: np.ndarray, mode: str):
    kh, kw = kernel.shape
    h, w = image.shape
    padded = np.zeros((h + 2 * (kh - 1), w + 2 * (kw - 1)))
    padded[kh - 1:kh - 1 + h, kw - 1:kw - 1 + w] = image
    if mode == "full":
        return padded, 0, 0, h + kh - 1, w + kw - 1
    if mode == "same":
        return padded, (kh - 1) // 2, (kw - 1) // 2, h, w
    raise ValueError(f"unknown mode {mode!r}")


def conv2d_four_loops(image: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    flipped = kernel[::-1, ::-1]
    kh, kw = flipped.shape
    padded, r0, c0, out_h, out_w = _pad_for(image, kernel, mode)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += flipped[u, v] * padded[r0 + i + u, c0 + j + v]
            out[i, j] = total
    return out


def conv2d_two_loops(image: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    flipped = kernel[::-1, ::-1]
    kh, kw = flipped.shape
    padded, r0, c0, out_h, out_w = _pad_for(image, kernel, mode)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[r0 + i:r0 + i + kh, c0 + j:c0 + j + kw] * flipped)
    return out


def conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return convolve2d(image, kernel, mode="same", boundary="symm")


def gaussian_kernel(sigma: float, ksize: int | None = None) -> np.ndarray:
    ksize = ksize or 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T


def gradients(image: np.ndarray):
    gx, gy = conv(image, DX), conv(image, DY)
    return gx, gy, np.sqrt(gx ** 2 + gy ** 2)


def orientation_hue(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    c, s = gx / magnitude, gy / magnitude
    root3 = np.sqrt(3) / 2
    wheel = np.dstack([c, -0.5 * c + root3 * s, -0.5 * c - root3 * s]) * 0.5 + 0.5
    return rgb_to_hsv(np.clip(wheel, 0, 1))[..., 0]


def orientation_image(gx: np.ndarray, gy: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    value = np.clip(magnitude / np.percentile(magnitude, 99.5), 0, 1)
    hsv = np.dstack([orientation_hue(gx, gy), np.ones_like(value), value])
    return hsv_to_rgb(hsv)
