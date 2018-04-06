import numpy as np
import matplotlib.pyplot as plt


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


def main():
    index = 121
    file_name = './data/weather/%d.RFLCTVTY' % index
    sweeps, metadata = read_reflectivity(file_name)

    sweep = 0
    plt.clf()
    plt.imshow(sweeps[sweep])
    plt.colorbar()
    plt.xlabel('angle')
    plt.ylabel('distance')
    plt.show()


if __name__ == '__main__':
    main()
