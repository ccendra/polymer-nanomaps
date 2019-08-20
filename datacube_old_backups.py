from skimage import io
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import os
import torch
import math
from scipy import ndimage
import masks as mask
import time
import pandas as pd


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
    fft = hrtem_fft_gpu[:, :, 0] ** 2 + hrtem_fft_gpu[:, :, 1] ** 2
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
    out = torch.cat((fft[-m // 2:], fft[:-m // 2]), dim=0)
    return torch.cat((out[:, -n // 2:], out[:, :-n // 2]), dim=1)


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
    # pad tensor with zeros function to get s x s tensor
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
    # get fft of padded tensor using torch.rfft function
    hrtem_fft_gpu = torch.rfft(pad(tensor), 2, normalized=True, onesided=False)
    # adding up real and imaginary components in FT
    fft = hrtem_fft_gpu[:, :, 0] ** 2 + hrtem_fft_gpu[:, :, 1] ** 2
    # shift zero frequency to center of image
    fft = tensor_shift_fft(fft)
    return fft


def median_filter(data, size=1, device='cuda'):
    """
    Median filter operation for n-dimension tensor or array. Function first checks if data is pyTorch tensor or numpy,
    then performs median filter operation using numpy and returns same datatype as input.
    Can be relatively slow operation because     only performed once per image/datacube.
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
        if row % 50 == 0:
            print('row: ', row)
        for j in range(N, n + 1, step_size):
            mini = normalize_tensor(img_gpu[i0:i, j0:j])
            window_mini = mini * hanning_window
            fft = tensor_fft(window_mini, s=NN)

            if plot:
                if ct % 20000 == 0:
                    fft_masked = fft * gaussian_ring
                    subplot_mini(mini.cpu(), window_mini.cpu(), fft.cpu(), fft_masked.cpu(), 'count = ' + str(ct))

            matrix[row, col, :] = get_orientation_torch(fft, filters_tensor, device)

            j0 += step_size
            ct += 1
            col += 1

        i0 += step_size
        row += 1

    print('Processing time to get 4D datacube [seconds]: ' + str(time.time() - start_time))
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
    :param img_datacube: 4D datacube of integrated powder spectrum. Torch tensor.
    :param df_mean_datacube: 4d datacube of  mean of the dark reference. Torch tensor.
    :param df_std_datacube: 4d datacube of standard deviation of the dark reference. Torch tensor.
    :param factor: how many std above mean dark reference intensity
    :param angles: list of angles used for the datacube analysis
    :return orientation_map: 2D pandas with orientation angle. NA if no peak, theta (degrees) otherwise.
    :return peak_intensity_tensor: 2D float tensor intensity of peaks at each (x,y, theta). Zero if no peak.
    """
    # Background threshold tensor 4D datacube. Background intensity at (x,y,th).
    threshold = df_mean_datacube + factor * df_std_datacube
    # Binary matrix of condition for peak or no peak.
    peak_condition_tensor = img_datacube > threshold.double()
    # 4D datacube with zeros if no peak, and intensity for thetas where there is peak.
    intensity_tensor = img_datacube * peak_condition_tensor.double()
    # Get value and index (theta) of maximum intensity. If there is no peak, the intensity is zero.
    peak_tensor, indices_tensor = torch.max(intensity_tensor.double(), dim=2)

    # Create dataframes to assign 'nan' to (x,y) locations where there is no peak
    peak_df = pd.DataFrame(peak_tensor.cpu().numpy()).astype('int')  # df of peak/no peak as 1s and 0s
    indices_df = pd.DataFrame(indices_tensor.cpu().numpy())[peak_df != 0]  # df with theta indices and nan for no peak

    start_angle = angles[0]  # to find orientation
    spacing = 180 / len(angles)
    orientation_map = start_angle + spacing * indices_df  # pandas df of theta in degrees orientation map

    return orientation_map, intensity_tensor


# Deprecated
def powder_spectrum(img, s):
    """Returns powder spectra by calculating 2D Fast Fourier Transform of one image.
    Args:
        img: image of size (x,y)
        s: input image for np.fft.fft2 function. If larger than
        image size, pads with zeros. s is size of output FFT.
    Returns:
        Amplitude squared of FFT
    """
    f = np.fft.fft2(img, s)
    return np.fft.fftshift(np.abs(f) ** 2)


def get_image(fn, threshold, n_frames, step):
    start_time = time.time()

    image_raw = open_tif(fn)[:n_frames, :, :]
    image_stack = stack_image(image_raw)
    image = find_bad_pixels(image_stack, threshold)

    x_crop = image.shape[0] % step
    y_crop = image.shape[1] % step
    image = image[x_crop // 2:-(x_crop // 2 + x_crop % 2), y_crop // 2:-(y_crop // 2 + y_crop % 2)]

    total_time = time.time() - start_time
    print('"Get Image" time (s): ' + str(np.round(total_time, 2)))
    return image


def normalize_image(image):
    img = image / np.mean(image) - 1
    return img.astype('float64')


def plot_image(image, title='', save_fn=''):
    """Plots image size (x,y). Re-scales image for larger plot.

    Args:
        image: 2D image of size (x,y)
        title: image title
        save_fn: filename if we want to save image
    """
    #     plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar()
    if save_fn != '':
        plt.savefig(save_fn, dpi=600)
    plt.show()


def subplot_mini(image, window_mini, fft_raw, fft_processed, title):
    """ Function plots real-space and FFT of image.

    Args:
        image: real-space image
        fft: FFT of image
    """
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    # plot TEM nanoimage with window
    plt.subplot(1, 4, 2)
    plt.imshow(window_mini, cmap='gray')
    plt.title(title)
    # plot FFT of nanoimage
    plt.subplot(1, 4, 3)
    plt.imshow(fft_raw, cmap='gray')
    plt.title('FFT raw')
    # plot FFT of nanoimage
    plt.subplot(1, 4, 4)
    plt.imshow(fft_processed, cmap='gray')
    plt.title('FFT masked')

    plt.show()


def find_bad_pixels(image, threshold):
    """Find really high intensity pixels corresponding
    artifacts. Bad pixels in scan and set them to zero.

    Args:
        image: numpy image of size (x,y)
        threshold: threshold intensity
    Returns:
        updated image
    """
    mask = image < threshold
    return np.multiply(mask, image)


def make_zeros_matrix_input(m, n, pixN):
    x = (int(m / pixN) + 1) * pixN
    y = (int(n / pixN) + 1) * pixN

    return np.zeros((x, y))


def center_fft(fft, center, width):
    x = center[0]
    y = center[1]
    return fft[x - width:x + width, y - width:y + width]


def fft_mask(fft_shape, lower_bound, upper_bound):
    m, n = fft_shape
    mask = np.zeros((m, n))
    center = np.array([m / 2, n / 2], dtype='int8')
    for i in range(m):
        for j in range(n):
            vector = [i, j] - center
            distance = np.linalg.norm(vector)
            if lower_bound < distance < upper_bound:
                mask[i, j] = 1
    return mask


def fft_slicing(fft_shape, upper_bound):
    m, n = fft_shape
    bound = int(upper_bound) + 1
    slicing = int(m / 2 - bound)
    length = 2 * int(bound) + 1
    return slicing, length


def get_masks():
    mask_list = []
    for file in os.listdir():
        if '_mask_angles' in file:
            mask_list.append(file)

    return mask_list


def get_nanofft(mini, s, mask, slicing):
    fft = powder_spectrum(mini, s)
    fft = np.multiply(fft, mask)
    crop_fft = fft[slicing:-slicing + 1, slicing:-slicing + 1]

    return crop_fft


def makeAzimuthalMask(q, theta, sigma_q, sigma_th, size):
    matrix = np.zeros((size, size))
    center = [matrix.shape[0] / 2, matrix.shape[1] / 2]

    x = np.linspace(0, matrix.shape[0], matrix.shape[0]) - center[0]
    y = np.linspace(0, matrix.shape[1], matrix.shape[1]) - center[1]

    x_gauss = np.exp(-(x + q) ** 2 / (2.0 * sigma_q ** 2))
    y_gauss = np.exp(-(y) ** 2 / (2.0 * sigma_th ** 2))

    matrix = 1 / (2 * math.pi * sigma_th * sigma_q) * np.outer(x_gauss, y_gauss.T)

    rotated = ndimage.rotate(matrix, theta, reshape=False)
    rotated = rotated + ndimage.rotate(rotated, 180, reshape=False)

    return rotated / np.sum(rotated)


def make_masks(pixN, thetas, sigma_th):
    size = 2 * pixN
    sigma_q = 1.8  # 1.23
    q = 6.25
    base_mask = np.zeros((size, size))

    masks_dic = {}

    for theta in thetas:
        mask = makeAzimuthalMask(q, theta, sigma_q, sigma_th, size=size)
        m, n = mask.shape
        base_mask[:m, :n] += mask
        masks_dic[theta] = mask

    #     plt.imshow(base_mask)
    #     plt.show()

    return masks_dic, base_mask


def get_scaling_factor(d, dx):
    """Function converts from real space to scaled reciprocal space
    Args:
        d: lattice spacing in Angstroms
        dx: calibrated Angstrom/pixel from microscope
    Returns:
        scaling: spatial frequency of feature (d) in inverse Angstrom / pixel units
    """
    q = 2 * math.pi / d
    d_pixels = d / dx
    q_pixels = d_pixels / 2

    scaling = q / q_pixels  # normalized spatial frequency

    return scaling


def process(img, pixN, step, masks_dic, base_mask, window, s, sigma5):
    start_time = time.time()

    m, n = img.shape
    i0 = 0
    ct = 0

    max_orientation = np.zeros((m // step, n // step))
    ct_row = 0

    for i in range(pixN, m + 1, step):
        j0 = 0
        ct_col = 0
        for j in range(pixN, n + 1, step):
            mini = normalize_image(img[i0:i, j0:j])
            mini = np.multiply(mini, window)

            fft_raw = powder_spectrum(mini, s)
            fft_processed = np.multiply(fft_raw, base_mask)

            if ct % 50000 == 0:
                subplot_mini(mini, fft_raw, fft_processed, 'count = ' + str(ct))

            angular_intensities = []
            theta_angles = []

            for key in masks_dic.keys():
                azimuthal_mask = masks_dic[key]
                normalization = np.sum(azimuthal_mask)

                angular_intensity = np.sum(np.multiply(azimuthal_mask, fft_processed)) / normalization

                if angular_intensity > sigma5:
                    angular_intensities.append(angular_intensity)
                    theta_angles.append(key)

            if angular_intensities != []:
                index = np.argmax(angular_intensities)
                max_orientation[ct_row, ct_col] = theta_angles[index]
            else:
                max_orientation[ct_row, ct_col] = -10

            j0 += step
            ct += 1
            ct_col += 1
        i0 += step
        ct_row += 1

    elapsed_time = time.time() - start_time

    print('done processing')
    print('elapsed time: ' + str(elapsed_time))

    return max_orientation


def process_terminal(img, pixN, step, masks_dic, base_mask, window, s, sigma5):
    start_time = time.time()

    m, n = img.shape
    i0 = 0
    ct = 0

    max_orientation = np.zeros((m // step, n // step))
    ct_row = 0

    for i in range(pixN, m + 1, step):
        j0 = 0
        ct_col = 0
        for j in range(pixN, n + 1, step):
            mini = normalize_image(img[i0:i, j0:j])
            mini = np.multiply(mini, window)

            fft_raw = powder_spectrum(mini, s)
            fft_processed = np.multiply(fft_raw, base_mask)

            if ct % 50000 == 0:
                print('count: ' + str(ct))

            angular_intensities = []
            theta_angles = []

            for key in masks_dic.keys():
                azimuthal_mask = masks_dic[key]
                normalization = np.sum(azimuthal_mask)

                angular_intensity = np.sum(np.multiply(azimuthal_mask, fft_processed)) / normalization

                if angular_intensity > sigma5:
                    angular_intensities.append(angular_intensity)
                    theta_angles.append(key)

            if angular_intensities != []:
                index = np.argmax(angular_intensities)
                max_orientation[ct_row, ct_col] = theta_angles[index]
            else:
                max_orientation[ct_row, ct_col] = -10

            j0 += step
            ct += 1
            ct_col += 1
        i0 += step
        ct_row += 1

    elapsed_time = time.time() - start_time

    print('done processing')
    print('elapsed time: ' + str(elapsed_time))

    return max_orientation


def get_orientation(fft, gaussian_q_filter, angles):
    intensity_theta = np.zeros((len(angles)))

    for i in range(len(angles)):
        intensity_theta[i] = np.sum(np.multiply(fft, ndimage.rotate(gaussian_q_filter, angles[i], reshape=False)))

    return intensity_theta