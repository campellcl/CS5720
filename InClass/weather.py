import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

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
                # TODO: convert to degress before using np.sin and np.cos.
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
    # fig.colorbar(im, cax=np.arange(start=np.min(values_thresholded, axis=0), stop=np.max(values_thresholded, axis=0)), orientation='horizontal')
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
    plot_circular_sweep_3d(sweep=sweeps[0], metadata=metadata[0], sweep_num=0)
    # plt.colorbar(mappable=colors)
    # plot_circular_sweeps(sweeps, metadata)

    # Goal: To create a scatter plot using only angle and distance/radius.
    # How to turn angle and distance into an x and y value?
    # Each sweep has a different angle and height associated with it. The angle is given by the azimuth
    # Elevation tells you


if __name__ == '__main__':
    main()
