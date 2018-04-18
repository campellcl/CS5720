from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import numpy as np
import math
from math import cos, sin

eye = None
target = None
up = None
fov_y = None
aspect = None
near = None
far = None
previous_point = None
window = None
button_down = None
vertices = []
normals = None
triangles = None
points = None
image_width = None
image_height = None
image_depth = None
win_id = None


def display():
    global eye, target, up, fov_y, aspect, near, far, vertices, points, normals

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov_y, aspect, near, far)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2],
              target[0], target[1], target[2],
              up[0], up[1], up[2])

    glLightfv(GL_LIGHT0, GL_POSITION, eye)

    color = [1.0, 1.0, 0.0, 1.]
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0)

    glDisable(GL_LIGHTING)
    glBegin(GL_TRIANGLES)
    for i in range(len(vertices)):
        glColor3fv([1, 1, 1])
        glNormal3fv(normals[i, :])
        glVertex3fv(vertices[i, :])
    glEnd()

    glPointSize(10)
    glBegin(GL_POINTS)
    for point, c in points:
        glColor3fv(c)
        glVertex3fv(point)
    glEnd()
    glEnable(GL_LIGHTING)
    glutSwapBuffers()


def motion_func(x, y):
    global win_id
    # this function modeled after modeler.PerspectiveCamera.orbit() function written by 'ags' here:
    # http://www.cs.cornell.edu/courses/cs4620/2008fa/asgn/model/model-fmwk.zip
    global previous_point, eye
    x *= 2 / window[0]
    y *= -2 / window[1]
    if button_down == 'left':
        mouse_delta = [x - previous_point[0], y - previous_point[1]]
        neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
        dist = sum([a**2 for a in neg_gaze]) ** (1/2)
        neg_gaze = [a / dist for a in neg_gaze]
        azimuth = math.atan2(neg_gaze[0], neg_gaze[2])
        elevation = math.atan2(neg_gaze[1], (neg_gaze[0]**2 + neg_gaze[2]**2)**(1/2))
        azimuth = (azimuth - mouse_delta[0]) % (2 * math.pi)
        elevation = max(-math.pi * .495, min(math.pi * .495, elevation - mouse_delta[1]))
        neg_gaze[0] = math.sin(azimuth) * math.cos(elevation)
        neg_gaze[1] = math.sin(elevation)
        neg_gaze[2] = math.cos(azimuth) * math.cos(elevation)
        mag = sum([a**2 for a in neg_gaze]) ** (1/2)
        neg_gaze = [a / mag * dist for a in neg_gaze]
        new_eye = [a + b for a, b in zip(target, neg_gaze)]
        eye = new_eye
        glutPostRedisplay()
    elif button_down == 'right':
        mouse_delta_y = y - previous_point[1]
        neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
        dist = sum([a**2 for a in neg_gaze]) ** (1/2)
        new_dist = dist * 2 ** (mouse_delta_y)
        new_neg_gaze = [a / dist * new_dist for a in neg_gaze]
        new_eye = [a + b for a, b in zip(target, new_neg_gaze)]
        eye = new_eye
        glutPostRedisplay()

    # print(eye)
    previous_point = (x, y)


def mouse_func(button, state, x, y):
    global previous_point, button_down
    # print(button_down, state, x, y)
    previous_point = (x * 2 / window[0], -y * 2 / window[1])
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            button_down = 'left'
        elif state == GLUT_UP:
            button_down = None
    elif button == GLUT_RIGHT_BUTTON:
        if state == GLUT_DOWN:
            button_down = 'right'
        elif state == GLUT_UP:
            button_down = None


def open_gl(sweeps, metadata, threshold=15):
    global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id, points
    image_width = len(sweeps)
    image_height = len(sweeps[0])
    image_depth = len(sweeps[0][0])
    eye = [(image_width-1)/2, (image_height-1)/2, 2*image_depth]
    target = [(image_width-1)/2, (image_height-1)/2, (image_depth-1)/2]
    up = [0, 1, 0]

    window = (800, 800)
    fov_y = 40
    near = .1
    far = 1000

    aspect = window[0] / window[1]
    light_position = eye
    light_color = [100.0, 100.0, 100.0, 1.0]

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window[0], window[1])
    win_id = glutCreateWindow(b'cubes')

    glClearColor(0., 0., 0., 1.)
    glShadeModel(GL_SMOOTH)
    # glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)

    glEnable(GL_PROGRAM_POINT_SIZE)

    glEnable(GL_LIGHT0)

    # Points:
    points = []
    x_coords = []
    y_coords = []
    z_coords = []
    values = []
    for n, sweep in enumerate(sweeps):
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
                values.append(sweeps[0][i][angle])
    # plt.clf()
    values_thresholded = []
    x_coords_thresholded = []
    y_coords_thresholded = []
    z_coords_thresholded = []
    # points.append(([1, 1, 1], [0, 0, 1]))
    for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
        if values[i] > threshold:
            values_thresholded.append(values[i])
            x_coords_thresholded.append(x)
            y_coords_thresholded.append(y)
            z_coords_thresholded.append(z)
            # print('appending point')
            points.append(([x, y, z], [0, 0, 1]))
    # points.append((list(zip(x_coords, y_coords)), [0, 0, 1]))
    # print(points)
    # # callbacks
    glutDisplayFunc(display)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)

    glutMainLoop()


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
    file_name = '../data/weather/%d.RFLCTVTY' % index
    sweeps, metadata = read_reflectivity(file_name)
    sweep = 0
    open_gl(sweeps, metadata[0], threshold=10)


if __name__ == '__main__':
    main()
