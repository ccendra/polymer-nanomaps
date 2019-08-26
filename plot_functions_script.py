from datacube import *
import masks as mask
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def fft(img, size, q_contour_list=[], color='blue', savefig='', linewidth=0.4):
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
            ax.add_patch(plt.Circle(((size-1) / 2, (size-1) / 2), q_pixels, facecolor='none',
                                    edgecolor=color, alpha=a, linewidth=linewidth, linestyle=':'))
            ax.annotate(str(np.round(q, 2)), xy=(size/2, size/2 + q_pixels), color=color, alpha=a, fontsize=8)

    if savefig:
        plt.savefig(savefig + '.png', transparent=False, dpi=600)
    #
    # ax.plot()  # Causes an auto scale update.
    # plt.show()
    plt.close()


def overlay_img_lines(img, lines, size, title, linewidth, linewidths=False, colors=False):
    fig, ax = plt.subplots(figsize=(size, size))
    if colors and linewidths:
        line_plot = mc.LineCollection(lines, linewidths=linewidths, colors=colors)
    elif colors:
        line_plot = mc.LineCollection(lines, linewidth=linewidth, colors=colors)
    else:
        line_plot = mc.LineCollection(lines, linewidth=linewidth)

    ax.add_collection(line_plot)
    plt.imshow(img)
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.title(title)
    plt.savefig(title + '.png', dpi=300, transparent=True)
    # plt.show()
    plt.close()


def hrtem(img, size=15, gamma=1, vmax=0, colorbar=True, savefig=''):
    """Plots HRTEM image in real space.
    Arguments:
        img: image numpy array. If using pyTorch tensor, must be send to cpu and converted to numpy.
        size: output figure size
        gamma: image contrast enhancer. Gamma = 1 as default (i.e no enhancement)
    """
    gamma_corrected = exposure.adjust_gamma(img, gamma)

    plt.figure(figsize=(size, size))
    if vmax:
        plt.imshow(gamma_corrected, cmap='gray', vmax=vmax)
    else:
        plt.imshow(gamma_corrected, cmap='gray')

    if colorbar:
        plt.colorbar()
    if savefig:
        plt.savefig(savefig + '.png', transparent=False, dpi=600)
    #
    # plt.show()
    plt.close()


def subplot_mini(image, window_mini, fft_raw, fft_processed, title):
    """ Plots stack of figures to describe nano-FFT extraction process. From left to right,
    the following figures are plot: real space  nano-image, windowed nano-image, raw FFT,
    and filtered FFT.
    Args:
        image: real-space image
        window_mini: real space image multiplied with Hanning window function
        fft_raw: calculated FFT
        fft_processed: FFT multiplied with Gaussian ring filter
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
    # plt.show()
    plt.close()

