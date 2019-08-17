from datacube import *
import masks as mask
from skimage import exposure


def fft(img, size, q_contour_list=[], color='blue'):
    """Plots Fourier transform and optionally radial contours of q space.
    Arguments:
        img: image numpy array. If using pyTorch tensor, must be send to cpu and converted to numpy.
        size: size of FFT
        q_contour_list: list of q values to be drawn as contours in figure
        color: color of contours
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')

    if len(q_contour_list) > 0:
        a = 0.5
        for q in q_contour_list:
            q_pixels = mask.get_q_pixels(q, size / 2)
            ax.add_patch(plt.Circle((size / 2, size / 2), q_pixels, facecolor='none',
                                    edgecolor=color, alpha=a, linewidth=1, linestyle=':'))
            ax.annotate(str(np.round(q, 1)), xy=(size/2, size/2 + q_pixels), color=color, alpha=a, fontsize=12)

    ax.plot()  # Causes an auto scale update.
    plt.show()


def hrtem(img, size=15, gamma=1):
    """Plots HRTEM image in real space.
    Arguments:
        img: image numpy array. If using pyTorch tensor, must be send to cpu and converted to numpy.
        size: output figure size
        gamma: image contrast enhancer. Gamma = 1 as default (i.e no enhancement)
    """
    gamma_corrected = exposure.adjust_gamma(img, gamma)

    plt.figure(figsize=(size, size))
    plt.imshow(gamma_corrected, cmap='gray')
    plt.colorbar()
    plt.show()