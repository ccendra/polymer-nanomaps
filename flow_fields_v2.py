import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import random
# from view_diffraction import *
# from matrix_masks import *
# from extras import *


class FlowLine:
    def __init__(self, row, col, theta, intensity):
        # Theta must be given in degrees
        self.index = (row, col)
        self.start_point = (row + 0.5, col + 0.5)  # Stay in coordinates of matrix. Rotate and flip later.
        self.x, self.y = self.start_point
        self.theta = theta
        self.start_theta = theta
        self.intensity = intensity
        self.edges = self.find_first_edges()
        self.lines = ([self.start_point, self.edges[0]],
                       [self.start_point, self.edges[1]])  # To seed lines moving in two different directions
        self.theta_lists = ([self.theta, self.theta], [self.theta, self.theta])
        self.intensity_lists = ([self.intensity, self.intensity], [self.intensity, self.intensity])

    def length(self):
        return len(self.lines[0]) + len(self.lines[1]) - 2

    def find_first_edges(self):
        # First check special cases
        if self.theta == 0 or self.theta == 180:
            return (np.floor(self.x), self.y), (np.ceil(self.x), self.y)
        elif self.theta == 90 or self.theta == 270:
            return (self.x, np.floor(self.y)), (self.x, np.ceil(self.y))
        else:
            # There are four possible intersections - one with each side of the box
            theta_rad = self.theta * np.pi / 180
            x_range = [np.floor(self.x), np.ceil(self.x)]
            y_range = [np.floor(self.y), np.ceil(self.y)]
            index = self.index
            center = self.start_point
            possible_points = [
                (index[0], center[1] + (index[0] - center[0]) * np.tan(theta_rad)),
                (index[0] + 1, center[1] + (index[0] + 1 - center[0]) * np.tan(theta_rad)),
                (center[0] + (index[1] - center[1]) / np.tan(theta_rad), index[1]),
                (center[0] + (index[1] + 1 - center[1]) / np.tan(theta_rad), index[1] + 1)
            ]
            # Due to symmetry, the nearest two points are the edge of the box.  45 degree angle crosses corner, which
            # results in a tie for closest which is fine as long as we get two different points.
            nearest_points = sorted(possible_points, key=lambda x: distance_2d(x, center))
            if self.theta == 45 or self.theta == 135:  # Remove Identical Points as needed
                while distance_2d(nearest_points[0], nearest_points[1]) < distance_2d(nearest_points[0], center):
                    nearest_points.remove(nearest_points[1])
                return nearest_points[:2]
            else:
                return nearest_points[:2]

    def propagate_2(self, peak_matrix, bend_tolerance=18):
        n_row, n_col, n_angles = peak_matrix.shape
        angle_step = 180 / n_angles

        # Zip parameters for the two directions of the line and loop to extend each line
        for line, theta_list, intensity_list in zip(self.lines, self.theta_lists, self.intensity_lists):
            while True:  # Loops until a 'break' statement occurs

                # Find row and column associated with the new box the line is entering
                current, old = (line[-1], line[-2])
                current_index = (int(np.floor(current[0])), int(np.floor(current[1])))  # Row, Col
                new_index = list(current_index)  # Using a list means we can change values one at a time
                direction = [0, 0]
                for dim in range(2): # Repeat these steps for x and for y
                    if current[dim] == int(current[dim]):  # Line is crossing into a new box
                        if current[dim] > old[dim]:
                            new_index[dim] += 0  # Already in the correct box
                            direction[dim] = 1
                        elif current[dim] < old[dim]:
                            new_index[dim] -= 1
                            direction[dim] = -1
                        else:
                            pass

                # Check if new point is inside the matrix
                if not (-1 < new_index[0] < n_row and -1 < new_index[1] < n_col):
                    # Out of range of matrix
                    # print('out of bounds')
                    break

                # Determine best angle for continuing the line
                old_theta = theta_list[len(line) - 1]
                possible_angles = [i*angle_step for i in range(n_angles) if peak_matrix[new_index[0], new_index[1], i]]
                if possible_angles:
                    delta_theta = [abs(i - old_theta) for i in possible_angles]
                    if min(delta_theta) <= bend_tolerance:
                        theta = possible_angles[np.argmin(delta_theta)]
                    else:
                        # Too much bending
                        # print('too much bending')
                        # line[-1] = []
                        # theta_list[-1] = []
                        break

                else:
                    # No available angles
                    # print('no available angles')
                    # line[-1] = []
                    # theta_list[-1] = []
                    break

                # Extend the line through the new box
                # First check special cases.
                if theta in [0, 180]:
                    new = (current[0] + direction[0], current[1])
                elif theta in [90, 270]:
                    new = (current[0], current[1] + direction[1])
                else:
                    # There are four possible intersections - one with each side of the box
                    theta_rad = theta*np.pi/180
                    possible_points = [
                        (new_index[0], current[1] + (new_index[0] - current[0]) * np.tan(theta_rad)),
                        (new_index[0] + 1, current[1] + (new_index[0] + 1 - current[0]) * np.tan(theta_rad)),
                        (current[0] + (new_index[1] - current[1]) / np.tan(theta_rad), new_index[1]),
                        (current[0] + (new_index[1] + 1 - current[1]) / np.tan(theta_rad), new_index[1] + 1)
                    ]

                    distances = [distance_2d(p, current) for p in possible_points]
                    farthest_points = sorted(possible_points, key=lambda x: distance_2d(x, current), reverse=True)

                    # Take the point that is the farthest away and is inside the box.  This means the starting
                    # point is only taken when necessary, and corners don't cause problems.
                    for p in farthest_points:
                        if new_index[0] <= p[0] <= new_index[0] + 1:  # Inside box
                            if new_index[1] <= p[1] <= new_index[1] + 1:  # Inside box
                                new = p
                                break
                    else:
                        # Couldn't propogate this way - all points out of the box?  Does that make sense?
                        # print('out of the box')
                        # print(distances)
                        break

                # Record results
                line.append(new)
                theta_list.append(theta)
                # intensity_list.append(intensity)


                # Enforce length limit to prevent infinite loops
                if len(line) > 50:
                    # print('too long')
                    break


def seed_lines(intensity_map, orientation_map, max_intensity, angles, maxes_only=True, min_spacing=2):
    """Returns list of seeds and numpy array with peak position as function of (x, y, angle).
    Arguments:
        intensity_map: (x,y,theta) numpy array datacube with intensity values
        orientation_map: numpy array with angle or 'nan' if no peak at gridpoint (x,y)
        max_intensity: numpy array with maximum intensity. Zero if no peak at gridpoint (x, y)
        angles: numpy list of angles
        maxes_only: boolean. Seed lines only for maximum intensity peak.
        min_spacing: Case of seeding lines for multiple peaks. Integer defines minimum angular separation (in terms of
        angular steps) present between two peaks.
    Returns:
        peaks_matrix: numpy array with ones denoting if there is a peak at each (x, y, theta)
        line_seeds: list of FlowLine collections
    """
    # Set up parameters
    n_rows, n_cols, n_angles = intensity_map.shape
    peaks_matrix = np.zeros((n_rows, n_cols, n_angles))
    line_seeds = []
    angle_step = int(180 / n_angles)

    # Generate peaks matrix
    for row in range(n_rows):
        for col in range(n_cols):
            # Case of only maximum intensity angle
            if maxes_only:
                intensity = max_intensity[row, col]
                if intensity > 0:
                    angle_index = int(orientation_map[row, col])
                    # Create new seed for a line and append to line_seeds
                    new = FlowLine(row, col, angles[angle_index], intensity)
                    line_seeds.append(new)
                    # Label presence of peak
                    peaks_matrix[row, col, angle_index] = 1
            # Case of multiple peaks
            elif min_spacing:
                for angle in range(n_angles):
                    # pad start and end with intensities
                    padded_matrix = np.pad(intensity_map[row, col, :], (n_angles, n_angles), 'wrap')
                    # select spacing matrix and find maximum, then append to list of seed lines.
                    spacing_matrix = padded_matrix[angle + n_angles - min_spacing: angle + n_angles + min_spacing]
                    if any(spacing_matrix) and intensity_map[row, col, angle] == np.max(spacing_matrix):
                        intensity = intensity_map[row, col, angle]
                        new = FlowLine(row, col, angle * angle_step, intensity)  # Angle in Degrees
                        peaks_matrix[row, col, angle] = 1
                        line_seeds.append(new)

    return line_seeds, peaks_matrix


def seed_lines_v2(intensity_map, maxes_only=True, min_spacing=5):
    """Returns a list of x,y,theta values"""
    # Set up variables
    n_rows, n_cols, n_angles = intensity_map.shape
    peaks_matrix = np.zeros((n_rows, n_cols, n_angles))
    line_seeds = []
    angle_step = int(180 / n_angles)

    # Subtract background and smooth matrix
    sub_matrix = intensity_map - np.min(intensity_map)

    # Generate peaks matrix
    for row in range(n_rows):
        for col in range(n_cols):
            for angle in range(n_angles):
                if sub_matrix[row, col, angle] > 0:
                    # Resolve maxes-only keyword if true
                    if maxes_only:
                        if sub_matrix[row, col, angle] == np.max(sub_matrix[row, col, :]):
                            # Otherwise, proceed to record the value of the peak
                            intensity = sub_matrix[row, col, angle]
                            new = FlowLine(row, col, angle * angle_step, intensity)  # Angle in Degrees
                            line_seeds.append(new)
                            # If you want to increase seeding density in the future, add a for loop here
                            peaks_matrix[row, col, angle] = 1
                    # Resolve min_spacing keyword if nonzero
                    # """
                    elif min_spacing:
                        padded_matrix = np.pad(sub_matrix[row, col, :], (n_angles, n_angles), 'wrap')
                        spacing_matrix = padded_matrix[
                                         angle + n_angles - min_spacing: angle + n_angles + min_spacing]
                        if any(spacing_matrix) and sub_matrix[row, col, angle] == np.max(spacing_matrix):
                            # Otherwise, proceed to record the value of the peak
                            intensity = sub_matrix[row, col, angle]
                            new = FlowLine(row, col, angle * angle_step, intensity)  # Angle in Degrees
                            peaks_matrix[row, col, angle] = 1
                            line_seeds.append(new)
                            # If you want to increase seeding density in the future, add a for loop here
                    # """
    return line_seeds, peaks_matrix


def color_by_angle(theta):
    radians = theta * 3.14 / 180
    red = np.cos(radians) ** 2  # Range is max 1 not max 255
    green = np.cos(radians + 3.14 / 3) ** 2 * 0.7  # Makes green darker - helps it stand out equally to r and b
    blue = np.cos(radians + 2 * 3.14 / 3) ** 2
    return (red, green, blue)


def trim_and_color(line_seeds, rotated_matrix, min_length, size, title, linewidth, window=False, window_shape = False, intensities=True):
    n_rows, n_col, n_angles = rotated_matrix.shape
    if window_shape:
        x1, x2, y1, y2 = window_shape
    # Sort lines by length
    random.shuffle(line_seeds)
    # ranked_lines = sorted(line_seeds, key = lambda x: x.length(), reverse=True)
    ranked_lines = sorted(line_seeds, key=lambda x: distance_2d(x.lines[0][-1], x.lines[1][-1]), reverse=True)
    random.shuffle(ranked_lines[:2000])

    # Create record-keeping matrix to track space-filling
    point_density = 2
    space_filling = np.zeros((n_rows * point_density, n_col * point_density, n_angles))

    # Trim lines to achieve desired visual spacing
    print('Trimming Lines...')
    position_spacing = 1
    theta_spacing = 15

    xy_spacing = position_spacing * point_density
    z_spacing = int(theta_spacing / (180 / n_angles))
    lines_to_print = []
    colors_to_print = []
    intensities_to_print = []
    short_lines = []
    short_line_colors = []
    intensities_to_print = []
    short_line_intensities = []
    intensity_matrix = subtract_min_as_background(rotated_matrix, 2)
    line_thickness_matrix = intensity_matrix ** 2 / np.percentile(intensity_matrix ** 2, 90)
    for seed in ranked_lines:
        backward = zip(seed.lines[0][0:], seed.theta_lists[0])
        forward = zip(seed.lines[1][0:], seed.theta_lists[1])
        for direction in (backward, forward):
            new_line = []
            for point, theta in direction:
                if new_line == []:
                    new_line.append(point)  # First point gets added automatically and doesn't take up space.
                else:
                    row = int(np.floor(point[0] * point_density)) - 1
                    col = int(np.floor(point[1] * point_density)) - 1
                    z = int(theta / (180 / n_angles))
                    ref_row = int(np.floor(point[0]) - 1)
                    ref_col = int(np.floor(point[1]) - 1)

                    if window:
                        if (ref_row < y1) or (ref_row >= y2) or (ref_col < x1) or (ref_col >= x2):
                            # print(y1, ref_row, y2)
                            # print(x1, ref_col, x2)
                            # Out of bounds of window image
                            break

                    if space_filling[row, col, z]:
                        # Space is taken
                        break
                    else:
                        # Space is available
                        new_line.append(point)
                        new_short_line = new_line[-2:]
                        short_lines.append(new_short_line)
                        short_line_colors.append(color_by_angle(theta))
                        try:  # Invesitgate this error more - row and column values outside grid
                            short_line_intensities.append(min(line_thickness_matrix[ref_row, ref_col, z], 2))
                        except IndexError:
                            raise
                            print(ref_row, ref_col, z)
                            short_line_intensities.append(0.5)
                        # Fill Space
                        if z - theta_spacing < 0:
                            space_filling[row, col, z - theta_spacing:] = 1
                            space_filling[row, col, :z + theta_spacing] = 1
                        elif z + theta_spacing >= n_angles - 1:
                            space_filling[row, col, z - theta_spacing:] = 1
                            space_filling[row, col, : z + theta_spacing - n_angles] = 1
                        else:
                            space_filling[row, col, z - theta_spacing:z + theta_spacing] = 1
            if len(new_line) >= min_length:
                lines_to_print = lines_to_print + short_lines
                short_lines = []
                colors_to_print = colors_to_print + short_line_colors
                short_line_colors = []
                intensities_to_print = intensities_to_print + short_line_intensities
                short_line_intensities = []
                # lines_to_print.append(new_line)
            else:
                short_lines = []
                short_line_colors = []
                short_line_intensities = []

    # lines_to_print.append([(1,60), (100, 60)])
    if window:
        lines_to_print.append([(y1, x1), (y1, x2), (y2, x2), (y2, x1), (y1, x1)])
    else:
        lines_to_print.append([(0, 0), (0, n_col), (n_rows, n_col), (n_rows, 0), (0, 0)])
    colors_to_print.append((0.4, 0.4, 0.6))

    if intensities:
        quick_line_plot(lines_to_print, size, title, linewidth, linewidths=intensities_to_print, colors=colors_to_print)
    else:
        quick_line_plot(lines_to_print, size, title, linewidth, colors=colors_to_print)
    #
    # plt.autoscale(enable=True, axis='y')
    # plt.autoscale(enable=True, axis='x')


def distance_2d(p1, p2):
    """Returns distance between two 2D vectors."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def quick_line_plot(lines, size, title, linewidth, linewidths=False, colors=False):
    """Plot collection of seeds as lines in a pyplot.
    Arguments:
        lines: collection of seed lines
        size: figure size
        title: plot title (optional)
        linewidth: width of lines
        linewidths: boolean asks if width of lines depends on intensity
        colors: boolean asks if using different colors for different lines. If not false, composed of list of colors.
    """
    fig, ax = plt.subplots(figsize=(size, size))
    if colors and linewidths:
        line_plot = mc.LineCollection(lines, linewidths=linewidths, colors=colors)
    elif colors:
        line_plot = mc.LineCollection(lines, linewidth=linewidth, colors=colors)
    else:
        line_plot = mc.LineCollection(lines, linewidth=linewidth)

    ax.add_collection(line_plot)
    ax.autoscale(enable=True, axis='both', tight=True)
    # plt.title(title)
    # plt.savefig(title + '.png', dpi=300, transparent=True)
    # plt.show()


def quick_line_plot_v2(lines, figsize, linewidth=0.2, linewidths=False, colors=False):
    fig, ax = plt.subplots(figsize=figsize)
    if colors and linewidths:
        line_plot = mc.LineCollection(lines, linewidths=linewidths, colors=colors)
    elif colors:
        line_plot = mc.LineCollection(lines, linewidth=linewidth, colors=colors)
    else:
        line_plot = mc.LineCollection(lines, linewidth=linewidth)
    ax.add_collection(line_plot)
    ax.autoscale(enable=True, axis='both', tight=True)


def preview_line_plot(line_seeds, size, title='', linewidth=0.2, **kwargs):
    """" Unpacks list of seeded lines and plots initial seeds.
    Arguments:
        line_seeds: list of seed class objects
        size: size of plot
        title: plot title (optional)
        linewidth: linewidth of drawn lines
        **kwargs: additional optional arguments
    """
    lines_to_print = []
    # Unpack seeds
    for seed in line_seeds:
        for line in seed.lines:
            lines_to_print.append(line)

    # Plot seeds
    quick_line_plot(lines_to_print, size, title, linewidth, *kwargs)


def preview_line_plot_v2(line_seeds, figsize, **kwargs):
    lines_to_print = []
    for seed in line_seeds:
        for line in seed.lines:
            lines_to_print.append(line)

    quick_line_plot_v2(lines_to_print, figsize, *kwargs)
    plt.autoscale(enable=True, axis='y')
    plt.autoscale(enable=True, axis='x')


def integrate_pi(image, pi_slice_masks):
    n_angles = len(pi_slice_masks)
    amplitudes = np.zeros(n_angles)
    for i,mask in enumerate(pi_slice_masks):
        # amplitudes[i] = np.sum(image[mask])
        amplitudes[i] = np.sum(image*mask)
    # Scale amplitudes to fit in hdf5 file
    return amplitudes


def normalize_1D(array, axis):
    # maxes = np.argmax(array, axis=axis)
    # normalized_array = np.divide(array, maxes)
    # new_array = np.divide(array, array.sum(axis=axis, keepdims=1))
    # maxes = array.sum(axis=axis, keepdims = 1)
    new_array = array/array.sum(axis=2, keepdims=1).astype(np.float)
    return new_array


def subtract_average_by_point(array, axis):
    # maxes = np.argmax(array, axis=axis)
    # normalized_array = np.divide(array, maxes)
    # new_array = np.divide(array, array.sum(axis=axis, keepdims=1))
    # maxes = array.sum(axis=axis, keepdims = 1)
    new_array = array - array.mean(axis=2, keepdims=1).astype(np.float)
    return new_array


def subtract_min_as_background(array, axis, verbose=False):
    new_array = array - array.min(axis=2, keepdims=1).astype(np.float)
    if verbose:
        # Verbose only works for this file for certain dimensions
        n_images = array.shape[2]
        for n in range(n_images):
            quick_heat_plot(array[:,:,n])
            quick_heat_plot(new_array[:,:,n])
    return new_array


# deprecated
# def seed_lines(raw_matrix, maxes_only=True, min_spacing=5):
#     """Returns a list of x,y,theta values"""
#     start_time = time.time()
#
#     # Set up variables
#     n_rows, n_cols, n_angles = raw_matrix.shape
#     peaks_matrix = np.zeros((n_rows, n_cols, n_angles))
#     line_seeds = []
#     angle_step = int(180 / n_angles)
#
#     # Subtract background and smooth matrix
#     # sub_matrix = subtract_min_as_background(raw_matrix, 2)
#     sub_matrix = raw_matrix
#
#     # Generate peaks matrix
#     for row in range(n_rows):
#         for col in range(n_cols):
#             for angle in range(n_angles):
#                 if sub_matrix[row, col, angle] > 0:
#                     # Resolve maxes-only keyword if true
#                     if maxes_only:
#                         if sub_matrix[row, col, angle] == np.max(sub_matrix[row, col, :]):
#                             # Otherwise, proceed to record the value of the peak
#                             intensity = sub_matrix[row, col, angle]
#                             new = FlowLine(row, col, angle * angle_step, intensity)  # Angle in Degrees
#                             line_seeds.append(new)
#                             # If you want to increase seeding density in the future, add a for loop here
#                             peaks_matrix[row, col, angle] = 1
#                     # Resolve min_spacing keyword if nonzero
#                     # """
#                     elif min_spacing:
#                         padded_matrix = np.pad(sub_matrix[row, col, :], (n_angles, n_angles), 'wrap')
#                         spacing_matrix = padded_matrix[
#                                          angle + n_angles - min_spacing: angle + n_angles + min_spacing]
#                         if any(spacing_matrix) and sub_matrix[row, col, angle] == np.max(spacing_matrix):
#                             # Otherwise, proceed to record the value of the peak
#                             intensity = sub_matrix[row, col, angle]
#                             new = FlowLine(row, col, angle * angle_step, intensity)  # Angle in Degrees
#                             peaks_matrix[row, col, angle] = 1
#                             line_seeds.append(new)
#                             # If you want to increase seeding density in the future, add a for loop here
#                     # """
#     print('processing time: ' + str(time.time() - start_time))
#
#     return line_seeds, peaks_matrix


# def compute_pi_intensity_matrix(dataset, q_min, q_max, q_per_pixel, angle_step):
#     n_images = dataset.attrs['n_images']
#     y_pixels = dataset.attrs['y_pixels']
#     x_pixels = dataset.attrs['x_pixels']
#     n_angles = int(180/angle_step)
#
#     # Generate masks
#     print('Generating Masks...')
#     pi_masks = pi_slice_masks(y_pixels, x_pixels, q_min, q_max, q_per_pixel, angle_step)
#
#     # Prepare output matrix
#     pi_matrix = np.zeros((n_images, n_angles))  # This is now 2D
#
#     # Integrate pi intensities
#     print('Integrating Intensities...')
#     progress_bar = ProgressBar(n_images)
#     for n in range(n_images):
#         image = np.array(dataset[:, :, n])
#         for i in range(n_angles):
#             mask = pi_masks[i]
#             if np.sum(mask) != 0:
#                 intensity = np.sum(np.multiply(image, mask)) / np.sum(mask)
#                 pi_matrix[n, i] = intensity
#         progress_bar.update(n)
#     return pi_matrix