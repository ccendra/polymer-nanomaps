import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


def search_peaks(img_datacube, df_mean_datacube, df_std_datacube, factor, angles):
    """
    Returns a list of x, y, theta values.
    :param img_datacube: 4D datacube of integrated powder spectrum. Torch tensor.
    :param df_mean_datacube: 4d datacube of  mean of the dark reference. Torch tensor.
    :param df_std_datacube: 4d datacube of standard deviation of the dark reference. Torch tensor.
    :param factor: how many std above?
    :param angles: list of angles used for the datacube analysis
    :return peak_tensor: 2D bytes tensor with peak or no peak at each (x, y)
    :return orientation_map: 2D pandas with orientation angle. NA if no peak, theta (degrees) otherwise.
    :return peak_intensity_tensor: 2D float tensor intensity of peaks at each (x,y, theta). Zero if no peak.
    """
    # Background threshold tensor 4D datacube. Background intensity at (x,y,th).
    threshold = df_mean_datacube + factor * df_std_datacube
    # Binary matrix of condition for peak or no peak.
    peak_condition_tensor = img_datacube > threshold.double()
    # 4D datacube with zeros if no peak, and intensity for thetas where there is peak.
    peak_intensity_tensor = img_datacube * peak_condition_tensor.double()
    # Get value and index (theta) of maximum intensity. If there is no peak, the intensity is zero.
    peak_tensor, indices_tensor = torch.max(peak_intensity_tensor.double(), dim=2)

    # Create dataframes to assign 'nan' to (x,y) locations where there is no peak
    peak_df = pd.DataFrame(peak_tensor.cpu().numpy()).astype('int')   # df of peak/no peak as 1s and 0s
    indices_df = pd.DataFrame(indices_tensor.cpu().numpy())[peak_df != 0]   # df with theta indices and nan for no peak

    start_angle = angles[0]   # to find orientation
    spacing = 180 / len(angles)
    orientation_map = start_angle + spacing * indices_df   # pandas df of theta in degrees orientation map

    return peak_tensor, orientation_map, peak_intensity_tensor



