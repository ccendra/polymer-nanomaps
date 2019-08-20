import numpy as np
from scipy import ndimage
import math


def circular_mask(n_pixels, q_center, q_bandwidth, dx):
    """Returns an array with a circular ring of ones on a field of zeros.
    This can be a slow step because it only happens once per batch.
    Method modified from Luke Bahlhorn.
    Args:
        n_pixels: number of pixels in y, x
        q_center: center of mask in q (1/A)
        q_bandwidth: bandwidth of mask in (1/A)
        dx: calibrated Angstrom/pixel from microscope
    Returns:
        mask: 2D circular mask
    """
    xc = n_pixels / 2 - 0.5
    yc = n_pixels / 2 - 0.5

    q_min = (q_center - q_bandwidth/2) * n_pixels/2
    q_max = (q_center + q_bandwidth/2) * n_pixels/2

    xv, yv = np.meshgrid(range(n_pixels), range(n_pixels))
    q_matrix = np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2) * dx
    mask = np.less_equal(q_min, q_matrix) * np.greater(q_max, q_matrix)

    return mask


def gaussian_q_filter(q, sigma_q, sigma_th, N, NN, dx):

    q_pixels = get_q_pixels(q, N)
    n = N * dx  # size of nano-image (in Angstrom)
    grid = np.linspace(-n / 2, n / 2, NN)  # Array centering

    sigma_q_pixels = sigma_q * N / 2  # bandwidth (inv nm)
    sigma_th_pixels = sigma_th * N / 2

    out = []

    for i in grid:
        a = 1 / (2 * math.pi * sigma_q_pixels * sigma_th_pixels)
        sub = ((grid - q_pixels) ** 2 / (2 * sigma_q_pixels ** 2) + (i) ** 2 / (2 * sigma_th_pixels ** 2))
        out.append(a * np.exp(-sub))

    matrix = np.array(out)
    matrix = matrix + ndimage.rotate(matrix, 180, reshape=False)

    return matrix


def get_q_pixels(q, N):
    return q * N/2


def gaussian_ring(angles, q, sigma_q, sigma_th, N, NN, dx):
    mask = gaussian_q_filter(q, sigma_q, sigma_th, N, NN, dx)

    for angle in angles:
        mask = mask + ndimage.rotate(mask, angle, reshape=False)
    return mask / np.amax(mask)