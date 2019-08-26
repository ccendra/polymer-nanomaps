from skimage import io
import numpy as np
import mrcfile
import torch
from scipy import ndimage
import masks as mask
import time
import pandas as pd
import plot_functions as plot
import math
import matplotlib.pyplot as plt


def read_tif(fn):
    """Opens raw .TIF image and returns numpy array.
    Args:
        fn: image filename
    Returns:
        np array of size (n, x, y)
        n is number of stacked images (generally 24)
        x, y is # pixels horizontally and vertically
    """
    img = io.imread(fn)

    return img.astype('float64')


def read_mrc(fn):
    """Opens .mrc file containing single stack of images and returns numpy array.
    Args:
        fn: image filename
    Returns:
        np array of size (n, x, y)
        n is number of stacked images (generally 24)
        x, y is # pixels horizontally and vertically
    """
    mrc = mrcfile.open(fn, mode='r')
    img = np.flip(mrc.data, axis=1)
    mrc.close()

    return img.astype('float64')


def stack_image(img_raw):
    """Returns sum of n stacked images
    Args:
        img_raw: np array of (n, x, y) OR (x,y) size
    Returns:
        np array size (x,y)
    """
    den = len(img_raw.shape)
    if den > 2:
        return np.sum(img_raw, axis=0)
    return img_raw


def normalize_tensor(tensor):
    """Tensor normalization operation. Tensor/mean - 1."""
    tensor = tensor / torch.mean(tensor) - 1
    return tensor


def tensor_fft(tensor, s=5000):
    """Returns powder spectra of 2D tensor (image) using PyTorch implementation.
    NOTE: location of operation (GPU or CPU) is determined by location of input tensor.
    Send tensor to GPU prior to using this function to perform operations in GPU (i.e. tensor.to(cuda))
    Args:
        tensor: 2D tensor (image)
        s: output size of FFT (s x s). tensor is padded with zeros prior to performing FFT operation
        to specified output size.
    Returns:
        fft: powder spectra (real^2 + complex^2) tensor of size (s x s) with Fourier Transform.
             DC frequency component is set in center of tensor.
    """
    m, n = tensor.shape
    # normalize tensor prior to performing FFT
    tensor = normalize_tensor(tensor)
    # pad tensor with zeros function to get (s x s) tensor
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
    # get fft of padded tensor using torch.rfft function
    hrtem_fft_gpu = torch.rfft(pad(tensor), 2, normalized=True, onesided=False)
    # adding up real and imaginary components in FT
    fft = hrtem_fft_gpu[:, :, 0]**2 + hrtem_fft_gpu[:, :, 1]**2
    # shift zero frequency to center of image
    fft = tensor_shift_fft(fft)
    return fft


def tensor_shift_fft(fft):
    """Shift zero frequency spatial frequency component to center of 2D image. For Pytorch implementation
    Args:
        fft: 2D FFT obtained using torch_fft function
    Returns:
        shifted FFT with DC frequency component in center of image.
    """
    m, n = fft.shape
    out = torch.cat((fft[-m//2:], fft[:-m//2]), dim=0)
    return torch.cat((out[:, -n//2:], out[:, :-n//2]), dim=1)


def median_filter(data, size=1, device='cuda'):
    """
    Median filter operation for n-dimension tensor or array. Function first checks if data is pyTorch tensor or numpy,
    then performs median filter operation using numpy and returns same datatype as input.
    Can be relatively slow operation because only performed once per image/datacube.
    :param data: torch tensor or numpy array to apply median filter to
    :param device: CUDA device
    :param size: size of sliding window. Default is size = 1
    :return: Median filtered tensor or numpy array
    """
    if type(data) == torch.Tensor:
        median_np = ndimage.median_filter(data.cpu().numpy(), size=size)
        return torch.from_numpy(median_np).to(device)
    else:
        return ndimage.median_filter(data, size=size)


def get_datacube(img_gpu, angles, step_size, q, out_fn, sigma_q, sigma_th, N, NN, device, dx, plot=False):
    """ Get intensity - theta 4D array. Saves 4D array output.
    Arguments:
        img_gpu: GPU tensor of raw image
        angles: np array with angles to probe
        step_size: size of steps during 'rastering'
        out_fn: filename of 4D datacube output numpy matrix
        q: peak center of lattice spacing features
        fn_map: output fn of map
        sigma_q: bandwidth in q of gaussian filter
        sigma_th: bandwidth in theta of gaussian filter
        N: Size of nano-image in pixels
        NN: size of FFT
        device: name of GPU CUDA device
        dx: pixel resolution
        plot: boolean to display subplots
    Returns:
        matrix: 4D pyTorch tensor containing integrated intesity for every (row, col, theta)
    """
    start_time = time.time()

    gaussian_q_filter = torch.from_numpy(mask.gaussian_q_filter(q, sigma_q, sigma_th, N, NN, dx))

    filters = np.zeros((NN, NN, len(angles)))
    for i in range(len(angles)):
        filters[:, :, i] = ndimage.rotate(gaussian_q_filter, angles[i], reshape=False)

    filters_tensor = torch.from_numpy(filters).to(device)

    size_rows = int((img_gpu.shape[0] - N) / step_size + 1)
    size_cols = int((img_gpu.shape[1] - N) / step_size + 1)
    matrix = torch.from_numpy(np.zeros((size_rows, size_cols, len(angles)))).to(device)

    hanning_window = torch.from_numpy(np.outer(np.hanning(N), np.hanning(N))).to(device)
    gaussian_ring = torch.from_numpy(mask.gaussian_ring(angles, q, sigma_q, sigma_th, N, NN, dx)).to(device)

    i0 = 0
    m, n = img_gpu.shape
    ct = 0
    row = 0

    for i in range(N, m + 1, step_size):
        j0 = 0
        col = 0
        # if row % 50 == 0:
            # print('row: ', row)
        for j in range(N, n + 1, step_size):
            mini = normalize_tensor(img_gpu[i0:i, j0:j])
            window_mini = mini * hanning_window
            fft = tensor_fft(window_mini, s=NN)

            if plot:
                if ct % 20000 == 0:
                    fft_masked = fft * gaussian_ring
                    plot.subplot_mini(mini.cpu(), window_mini.cpu(), fft.cpu(), fft_masked.cpu(), 'count = ' + str(ct))

            matrix[row, col, :] = get_orientation_torch(fft, filters_tensor, device)

            j0 += step_size
            ct += 1
            col += 1

        i0 += step_size
        row += 1

    # print('Processing time to get 4D datacube [seconds]: ' + str(time.time() - start_time))
    np.save(out_fn, matrix.cpu().numpy())

    return matrix


def get_orientation_torch(fft, filters, device):
    """ Gets Intensity tensor for different angles at any grid point (x,y). Uses broadcasting and torch
    operations to speed 2x process with respect to loop.
    :param fft: GPU torch tensor of fourier transform
    :param filters: GPU torch tensor of gaussian filters to be applied (different angles)
    :param device: CUDA device
    :return: intensity tensor at grid point x,y
    """
    m, n = fft.shape
    fft_broadcast = torch.empty(m, n, 1).to(device).double()
    fft_broadcast[:, :, 0] = fft

    intensity_theta = torch.sum(torch.mul(fft_broadcast, filters), [0, 1])

    return intensity_theta


def search_peaks(img_datacube, df_mean_datacube, df_std_datacube, factor, angles):
    """
    Returns a list of x, y, theta values.
    Arguments:
        img_datacube: 4D datacube of integrated powder spectrum. Torch tensor.
        df_mean_datacube: 4d datacube of  mean of the dark reference. Torch tensor.
        df_std_datacube: 4d datacube of standard deviation of the dark reference. Torch tensor.
        factor: how many std above mean dark reference intensity
        angles: list of angles used for the datacube analysis
    Returns:
        indices_df: 2D pandas with orientation angle index. NA if no peak, theta (degrees) otherwise.
        intensity_tensor: 3D float tensor intensity of peaks at each (x,y, theta). Zero if no peak.
        max_intensity_df: 2D pandas with maximum value of intensity. Zero if no peak.
    """
    # Background threshold tensor 4D datacube. Background intensity at (x,y,th).
    threshold = df_mean_datacube + factor * df_std_datacube
    # Binary matrix of condition for peak or no peak.
    peak_condition_tensor = img_datacube > threshold.double()
    # 4D datacube with zeros if no peak, and intensity for thetas where there is peak.
    intensity_tensor = img_datacube * peak_condition_tensor.double()
    # Get value and index (theta) of maximum intensity. If there is no peak, the intensity is zero.
    max_intensity_tensor, indices_tensor = torch.max(intensity_tensor.double(), dim=2)

    # Create dataframes to assign 'nan' to (x,y) locations where there is no peak
    max_intensity_df = pd.DataFrame(max_intensity_tensor.cpu().numpy()).astype('int')   # df of peak/no peak as 1s and 0s
    indices_df = pd.DataFrame(indices_tensor.cpu().numpy())[max_intensity_df != 0]   # df with theta indices and nan for no peak
    indices_df.head()

    # start_angle = angles[0]   # to find orientation
    # spacing = 180 / len(angles)
    # orientation_map_df = start_angle + spacing * indices_df   # pandas df of theta in degrees orientation map

    return indices_df, intensity_tensor, max_intensity_df


