import torch
import numpy as np
import extras as ex


def calculate_all_correlations(z, device='cuda'):
    """Calculates all possible one point vs. all points correlations. Function uses torch tensors and outer product
    to accelerate computations. Every iteration produces the correlations between one point vs. all points, then stores
    them in a matrix containing sum of all correlations and arranged as function of relative distance in a 2D map.
    Arguments:
        z: input datacube tensor of size (m, n, th)
        device: CUDA device
    Returns:
        corr_values: torch tensor (2m, 2n, th) with the sum of all possible correlations placed as function of
        relative (m, n, th) location
        count_values: torch tensor (2m, 2n) with total number of times a certain correlation at each grid point was
        calculated. Tensor allows calculating the mean correlation.

    """
    m, n, th = z.shape

    print('...calculating correlations...')
    progress_bar = ex.ProgressBar(n)

    corr_values = torch.zeros(2 * m, 2 * n, th, dtype=torch.double).to(device)
    count_values = torch.zeros(2 * m, 2 * n, 1).to(device)

    for col in range(n):
        #     print('    col: ', col)
        for row in range(m):
            zi = z[row, col, :]
            z_1d = torch.reshape(z, (-1,))
            op_raw = torch.ger(z_1d, zi)
            op_sum = torch.sum(torch.reshape(op_raw, (m, n, th, th)), dim=2) / th

            corr_values[(m - row): (2 * m - row), (n - col): (2 * n - col), :] += op_sum
            count_values[(m - row): (2 * m - row), (n - col): (2 * n - col)] += 1
        progress_bar.update('+1')

    return corr_values, count_values


def sort_correlations(corr_values, count_values, z, threshold=100, device='cuda'):
    """Sorts calculated correlations as function of spatial and angular distance. """
    m, n, th = z.shape
    print('...sorting correlations as a function of distance...')

    r_values = np.array(list(set(np.reshape(map_r_values(m, n), -1))), dtype='double')
    r_values.sort()

    r_map = make_r_map(m, n)

    correlations = torch.zeros(len(r_values), th).to(device)
    distance = torch.zeros(len(r_values), 1).to(device)

    threshold = threshold * th
    i = 0
    for r in r_values:
        counts = torch.sum(count_values[r_map == r])
        if counts > threshold:
            correlations[i, :] = torch.sum(corr_values[r_map == r], dim=0) / counts.double()
            distance[i] = r
            i += 1

    z_mean = torch.mean(z).float()

    correlations = (correlations[:i, :] / z_mean ** 2).cpu().numpy()
    distance = distance[:i, 0].cpu().numpy()

    print('...correlations have been sorted as function of distance and orientation')

    return correlations, distance


def calculate_2d_distance(a, b):
    """Returns distance between two pixels"""
    return np.round(np.sqrt(a ** 2 + b ** 2), 4)


def calculate_distance_array(m, n):
    """Generates numpy array with log of distances between pixels. Reference is the top left corner of image
    Args:
        m, n: image
    Returns:
        distance: (m x n, 1) numpy array with distances. Values rounded to second decimal point.
    """
    d = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d[i, j] = calculate_2d_distance(i, j)

    return np.round(d.reshape((m * n, 1)), 2)


def th_weights_matrix(n_angles):
    """Generates spatial weight matrix for theta differences. This is possible because the order in
    terms of theta is always constant.
    Args:
        n_angles: integer number of angles
    Returns:
        w_th: numpy array of size (n_angles, n_angles, n_angles, 1) with ones for points with equal delta_theta.
        Dim=1 corresponds to delta_theta.
    """
    w_th = np.zeros((n_angles, n_angles, n_angles, 1))
    delta_th = np.linspace(0, n_angles - 1, n_angles)
    for th1 in range(n_angles):
        for th2 in range(n_angles):
            for d_th in delta_th:
                idx = int(np.abs(th1 - th2))
                if idx == d_th:
                    w_th[idx, th1, th2, 0] = 1
    return w_th


def map_r_values(m, n):
    """Creates 2D numpy array of distance values at each (x, y) grid point for specified 2D array size.
    Arguments:
        m: number of rows
        n: number of columns
    Returns:
        out: numpy array of size (m, n) with distance values relative to top left corner of array (i.e. point (0,0)).
            Values are rounded to first decimal point.
    """
    x = np.arange(m)
    out = np.zeros((m, n))
    for col in range(n):
        out[:, col] = np.sqrt(x**2 + col**2)

    out = np.round(out, 1)
    return out


def make_r_map(m, n, device='cuda'):
    """Creates 2D torch tensor with distance values relative to center of the array. We use this tensor to quickly
    sort correlation values as a function of distance.
    Args:
        m: number of rows
        n: number of columns
        device: CUDA device
    Returns:
        out: torch tensor of size (2m, 2n) with distance values relative to a center point.
    """
    r_map = map_r_values(m, n)
    out = np.zeros((2 * m, 2 * n))
    out[1:m + 1, 1:n + 1] = np.flip(r_map)
    out[m:, n:] = r_map
    out[1:m + 1, n:] = np.flip(r_map, axis=0)
    out[m:, 1:n + 1] = np.flip(r_map, axis=1)

    out = torch.from_numpy(out).to(device)
    return out