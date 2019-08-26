from datacube import *
import masks as mask
import flow_fields as ff
import plot_functions_script as plot
import os
import time


# Pytorch setup
cuda = torch.device('cuda')
torch.cuda.get_device_name(cuda)


def create_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)


def assemble_filename(pwd, base_name, ID, ending='.mrc'):
    return pwd + base_name + str(ID).rjust(5, '0') + ending


def process_datacube(output_path, pwd, ID, base_name, stack_range, angles, step_size,  q_contour, q,
                     sigma_q, sigma_th, percentile_factor, N, NN, dx, img_size=-1, keyword='', exception=False):
    # Create folder for outputs
    folder_name = output_path + str(ID).rjust(5, '0')
    create_directory(folder_name)

    # Import HRTEM image
    print('importing hrtem image')
    fn = assemble_filename(pwd, base_name, ID, ending='.mrc')
    data = read_mrc(fn)

    img_np = stack_image(data[:stack_range, :, :])
    if exception:
        img_np = img_np[500:3200, :2700]

    if img_size != -1:
        img_np = img_np[-img_size:, -img_size:]

    # get hrtem image tensor.
    # NOTE: tensor and numpy share the same memory in CPU!
    print('.............sending tensors to GPU')
    img_tensor = torch.from_numpy(img_np)
    img_tensor_gpu = img_tensor.to(cuda)  # tensor send to gpu not related in memory to numpy

    plot.hrtem(img_np, size=10, gamma=0.4, vmax=20, savefig=folder_name + '/hrtem' + keyword, colorbar=False)

    img_fft_gpu = tensor_fft(normalize_tensor(img_tensor_gpu), s=6000)
    plot.fft(np.log(img_fft_gpu.cpu()) ** 2, 6000, q_contour, savefig=folder_name + '/fft_contoured' + keyword)
    plot.fft(np.log(img_fft_gpu.cpu()) ** 2, 6000, [], savefig=folder_name + '/fft' + keyword)

    # gaussian_q_filter = mask.gaussian_q_filter(q, sigma_q, sigma_th, N, NN, dx)

    print('.............getting datacube')
    img_datacube = get_datacube(img_tensor_gpu, angles, step_size, q, folder_name + '/datacube' + keyword + '.npy', sigma_q, sigma_th,
                                N, NN, device='cuda', dx=dx, plot=False)
    img_datacube = median_filter(img_datacube, device=cuda, size=8)

    print('.............loading dark reference')
    pwd_dr = '/home/camila/PycharmProjects/2019_may/dark_references/'
    m, n, k = img_datacube.shape
    background_mean_tensor = torch.mean(torch.from_numpy(np.load(pwd_dr + 'df_angular_mean.npy')).to(cuda)[:m, :n, :],
                                        dim=2, keepdim=True)
    background_std_tensor = torch.mean(torch.from_numpy(np.load(pwd_dr + 'df_angular_std.npy')).to(cuda)[:m, :n, :],
                                       dim=2, keepdim=True)

    print('.............extracting info from datacube')
    orientation_map_df, intensity_map_tensor, max_intensity_df = search_peaks(img_datacube, background_mean_tensor,
                                                                              background_std_tensor, percentile_factor,
                                                                              angles)
    np.save(folder_name + '/intensity_map' + keyword + '.npy', ndimage.rotate(intensity_map_tensor.cpu().numpy(), -90))

    plt.figure(figsize=(8, 8))
    plt.imshow(orientation_map_df)
    plt.colorbar()
    plt.savefig(folder_name + '/orientation_map' + keyword + '.png', transparent=False, dpi=600)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(pd.DataFrame(max_intensity_df)))
    plt.colorbar()
    plt.savefig(folder_name + '/maximum_intensity' + keyword + '.png', transparent=False, dpi=600)
    plt.close()


def draw_flow_lines(output_path, ID, linewidth, min_spacing, keyword='', size=10):

    start_time = time.time()
    folder_name = output_path + str(ID).rjust(5, '0')

    print('Loading Image ID: ' + str(ID))
    matrix = np.load(folder_name + '/intensity_map' + keyword + '.npy')

    print('.............seeding flow lines')
    line_seeds, peak_matrix = ff.seed_lines(matrix, maxes_only=False, min_spacing=min_spacing)
    ff.preview_line_plot(line_seeds, size, linewidth)
    directors = plt.gcf()
    directors.savefig(folder_name + '/seed_flow_lines' + keyword + '.png', transparent=False, dpi=600)
    plt.close()

    print('.............propagating flow lines')
    for line in line_seeds:
        line.propagate_2(peak_matrix, bend_tolerance=10)

    ff.preview_line_plot(line_seeds, size, linewidth)
    propagated = plt.gcf()
    propagated.savefig(folder_name + '/propagated_flow_lines' + keyword + '.png', transparent=False, dpi=600)
    plt.close()

    print('.............trimming and coloring lines')
    min_length = 2  # Must pass outside origninal box to be drawn
    window_shape = (20, 60, 20, 60)
    window = False
    ff.trim_and_color(line_seeds, matrix, min_length, size, window=window, window_shape=window_shape, intensities=True)
    flow_field = plt.gcf()
    flow_field.savefig(folder_name + '/flow_field_weighted' + keyword + '.png', transparent=False, dpi=600)
    plt.close()

    print('.............unweighted trimming and coloring lines')
    ff.trim_and_color(line_seeds, matrix, min_length, size, window=window, window_shape=window_shape, intensities=False)
    unweighted_flow_field = plt.gcf()
    unweighted_flow_field.savefig(folder_name + '/flow_field_unweighted' + keyword + '.png', transparent=False, dpi=600)
    plt.close()

    print('processing time: ' + str(np.round(time.time() - start_time, 2)))