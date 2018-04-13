import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
import copy
from InClass.open_gl_framework import OpenGLFramework


def read_reflectivity(file_name):
    sweeps = []
    metadata = []
    with open(file_name, 'rb') as fp:
        for sweep in range(1, 10):
            # read sweep delimiter
            line = fp.readline().strip().decode('utf-8')
            header = 'SWEEP%dRFLCTVTY' % sweep
            if line != header:
                print('Error: Failed to find "%s" in "%s"' % (header, line))
                return

            # print('Sweep %d' % sweep)

            # read latitude, longitude, height
            line = fp.readline().strip().decode('utf-8')
            # print(line)
            tokens = line.split()
            if len(tokens) != 6 or tokens[0] != 'Latitude:' or tokens[2] != 'Longitude:' or tokens[4] != 'Height:':
                print('Error: Failed to find Lat, Lon, Ht in %s' % tokens)
                return
            latitude = float(tokens[1])
            longitude = float(tokens[3])
            height = float(tokens[5])
            # print('lat', latitude, 'lon', longitude, 'height', height)

            # read number of radials
            num_radials = int(fp.readline().strip().decode('utf-8'))
            # print(num_radials, 'radials')

            gate_dist = float(fp.readline().strip().decode('utf-8'))
            # print(gate_dist, 'meters to gate')

            sweep_data = {
                'latitude': latitude,
                'longitude': longitude,
                'height': height,
                'num_radials': num_radials,
                'gate_dist': gate_dist
            }

            data = []
            radial_data = []
            for radial in range(num_radials):
                # print('for radial %d out of %d' % (radial, num_radials))
                tokens = fp.readline().strip().split()
                current_radial, num_gates, gate_width = (int(t) for t in tokens[:3])
                beam_width, azimuth, elevation = [float(t) for t in tokens[3:-1]]
                start_time = int(tokens[-1])
                # print(current_radial, num_gates, gate_width, beam_width, azimuth, elevation, start_time)
                empty_line = fp.readline().strip().decode('utf-8')
                if empty_line != '':
                    raise (Exception('Error: no empty line'))

                seconds_since_epoch = fp.readline().strip().decode('utf-8')
                if seconds_since_epoch != 'seconds since epoch':
                    raise (Exception('Error: no "seconds since epoch"'))

                x = np.fromfile(fp, dtype='>f', count=num_gates)
                x[x < 0] = 0
                data.append(x)
                radial_data.append({
                    'beam_width': beam_width,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'start_time': start_time,
                })
            data = np.array(data)
            data = data.T
            sweeps.append(np.array(data))
            metadata.append({
                'sweep': sweep_data,
                'radials': radial_data
            })

        sweeps = np.array(sweeps)
        for i in range(len(sweeps)):
            print('sweep %d: [%g, %g], %g +/- %g' % (
                i, sweeps[i].min(), sweeps[i].max(), sweeps[i].mean(), sweeps[i].std()))

    return sweeps, metadata


def plot_circular_sweep_2d(sweep, sweep_num, colors):
    x_coords = []
    y_coords = []
    values = []
    for i, distances in enumerate(sweep):
        for angle, distance in enumerate(distances):
            # print(sweep[i][angle])
            # TODO: Convert from degrees to radians before using np.cos and np.sin
            x = np.cos(np.radians(angle))*i
            y = np.sin(np.radians(angle))*i
            x_coords.append(x)
            y_coords.append(y)
            values.append(sweep[i][angle])
    plt.clf()
    plt.scatter(x_coords, y_coords, c=values)
    plt.title('Sweep %d' % sweep_num)
    plt.colorbar()
    plt.show()


def plot_circular_sweeps(sweeps, metadata):
    x_coords = []
    y_coords = []
    for n, sweep in enumerate(sweeps):
        for i, distances in enumerate(sweep):
            # TODO: Should really be using the azumith in the metadata for angle.
            for angle, distance in enumerate(distances):
                x = np.cos(np.radians(angle))*distance
                y = np.sin(np.radians(angle))*distance
                x_coords.append(x)
                y_coords.append(y)
    plt.clf()
    plt.scatter(x_coords, y_coords)
    plt.title('All Sweeps')
    # plt.colorbar()
    plt.show()


def plot_circular_sweep_3d(sweep, metadata, sweep_num, threshold=10):
    x_coords = []
    y_coords = []
    z_coords = []
    values = []
    for i, distance in enumerate(sweep):
        for angle, distance in enumerate(distance):
            # height = metadata['sweep']['height']
            elevation = metadata['radials'][angle]['elevation']
            x = np.cos(np.radians(angle))*i
            y = np.sin(np.radians(angle))*i
            z = i * np.sin(np.radians(elevation))
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            values.append(sweep[i][angle])
    # plt.clf()
    values_thresholded = []
    x_coords_thresholded = []
    y_coords_thresholded = []
    z_coords_thresholded = []
    for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
        if values[i] > threshold:
            values_thresholded.append(values[i])
            x_coords_thresholded.append(x)
            y_coords_thresholded.append(y)
            z_coords_thresholded.append(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x_coords_thresholded, y_coords_thresholded, z_coords_thresholded, c=values_thresholded)
    plt.title('Sweep %d 3D' % sweep_num)
    plt.colorbar(mappable=im, ax=ax)
    plt.show()


def get_x_y_z_values_from_sweep(sweep, metadata):
    x_coords = []
    y_coords = []
    z_coords = []
    values = []
    for i, distance in enumerate(sweep):
        for angle, distance in enumerate(distance):
            # height = metadata['sweep']['height']
            elevation = metadata['radials'][angle]['elevation']
            x = np.cos(np.radians(angle))*i
            y = np.sin(np.radians(angle))*i
            z = i * np.sin(np.radians(elevation))
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            values.append(sweep[i][angle])
    return x_coords, y_coords, z_coords, values


# def marching_squares_contour(sweep, sweep_num):
#     """
#     marching_squares_contour: Divides the provided sweep data into contour cells and populates with contour lines using
#         the marching squares algorithm.
#     :param sweep: A single doppler radar sweep from a .RFLCTVTY file.
#     :return contours: A list of lists (2d array) containing the start and end points for each contour line segment.
#     """
#     lookup_table = {
#         '0000': None,
#         '0001': [(0, 0.5), (0.5, 1.0)],
#         '0010': [(0.5, 1), (1.0, 0.5)],
#         '0011': [(0, 0.5), (1.0, 0.5)],
#         '0100': [(0.5, 0), (1.0, 0.5)],
#         '0101': [(0, 0.5), (0.5, 0), (0.5, 1.0), (1.0, 0.5)],
#         '0110': [(0.5, 1.0), (0.5, 0)],
#         '0111': [(0, 0.5), (0.5, 0)],
#         '1000': [(0, 0.5), (0.5, 0)],
#         '1001': [(0.5, 1.0), (0.5, 0)],
#         '1010': [(0, 0.5), (0.5, 1.0), (0.5, 0), (1.0, 0.5)],
#         '1011': [(0.5, 0), (1.0, 0.5)],
#         '1100': [(0, 0.5), (1.0, 0.5)],
#         '1101': [(0.5, 1.0), (1.0, 0.5)],
#         '1110': [(0, 0.5), (0.5, 1.0)],
#         '1111': None
#     }
#
#     x_coords = []
#     y_coords = []
#     values = []
#     for i, distances in enumerate(sweep):
#         for angle, distance in enumerate(distances):
#             # print(sweep[i][angle])
#             # TODO: Convert from degrees to radians before using np.cos and np.sin
#             x = np.cos(np.radians(angle))*i
#             y = np.sin(np.radians(angle))*i
#             x_coords.append(x)
#             y_coords.append(y)
#             values.append(sweep[i][angle])
#     plt.clf()
#     plt.scatter(x_coords, y_coords, c=values)
#     plt.title('Sweep %d' % sweep_num)
#     plt.colorbar()
#
#
#
#     plt.show()
#
#
#
#     contours = []
#     contour_cells = []
#     for i in range(len(sweep) - 1):
#         contour_cell_row = []
#         for j in range(len(sweep[i]) - 1):
#             contour_cell = [(i, j), (i, j+1), (i+1, j+1), (i+1, j)]
#             contour_cell_row.append(contour_cell)
#             # Create a binary index based on the values in the cell:
#             cell_bin_index = ''.join('1' if sweep[x, y] > 0 else '0' for x, y in contour_cell)
#             # Perform a lookup using the binary index:
#             contour_line_segments = lookup_table[cell_bin_index]
#             contour_segment = []
#             if contour_line_segments is not None:
#                 # If there are two lines present...
#                 if len(contour_line_segments) == 4:
#                     for n, (x, y) in enumerate(contour_line_segments):
#                         contour_segment.append((x+j, y+i))
#                         if n == 1:
#                             contours.append(contour_segment)
#                             contour_segment = []
#                         elif n == 3:
#                             contours.append(contour_segment)
#                 else:
#                     for n, (x, y) in enumerate(contour_line_segments):
#                         contour_segment.append((x+j, y+i))
#                     contours.append(contour_segment)
#         contour_cells.append(contour_cell_row)
#     return contours


def plot_contour_lines_2d(x_coords, y_coords, values, sweep_num, contour_lines):
    fig, ax = plt.subplots()
    plt.scatter(x_coords, y_coords, c=values)
    lc = mc.LineCollection(contour_lines, linewidths=2, color='red', linestyles='solid')
    ax.add_collection(lc)
    plt.title('Sweep %d' % sweep_num)
    plt.colorbar()
    plt.show()


def main():
    index = 121
    file_name = '../data/weather/%d.RFLCTVTY' % index
    sweeps, metadata = read_reflectivity(file_name)
    sweep = 0
    # plt.clf()
    # plt.imshow(sweeps[sweep])
    # plt.colorbar()
    # plt.xlabel('angle')
    # plt.ylabel('distance')
    # plt.show()
    # map values to colors:
    colors = sweeps[0].reshape((-1, ))
    # Plot one sweep in 2d:
    # plot_circular_sweep_2d(sweep=sweeps[0], sweep_num=0, colors=colors)
    # Plot one sweep in 3d:
    # plot_circular_sweep_3d(sweep=sweeps[0], metadata=metadata[0], sweep_num=0)
    x, y, z, values = get_x_y_z_values_from_sweep(sweeps[0], metadata[0])
    # OpenGLFramework(x=x, y=y, z=z, values=values)
    x = np.array(x).reshape((int(len(x)/2), -1))
    y = np.array(y).reshape((int(len(y)/2), -1))
    z = np.array(z).reshape((int(len(z)/2), -1))
    CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    # contour_lines = marching_squares_contour(sweep=sweeps[0])
    # plot_contour_lines_2d(x_coords=x, y_coords=y, values=values, sweep_num=0, contour_lines=contour_lines)
    # plt.colorbar(mappable=colors)
    # plot_circular_sweeps(sweeps, metadata)
    pass
    # Goal: To create a scatter plot using only angle and distance/radius.
    # How to turn angle and distance into an x and y value?
    # Each sweep has a different angle and height associated with it. The angle is given by the azimuth
    # Elevation tells you


if __name__ == '__main__':
    main()
