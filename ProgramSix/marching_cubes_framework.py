"""
marching_cubes_framework.py
Debugging parameters: ./data/cubes/001 2 1
"""

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import math
import numpy as np
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


def make_cubes():
    grid_size = 16
    x = np.zeros((grid_size * 3, grid_size * 3, 2))
    m, n, p = x.shape

    ii = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    jj = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    kk = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    for h in range(256):
        i = (h % 16) * 3
        j = int(h / 16) * 3
        x[ii+i, jj+j, kk] = [int(a)*255 for a in list('{0:08b}'.format(h))]

    for k in range(p):
        file_name = './data/cubes/%02d.pgm' % (k + 1)
        with open(file_name, 'w') as fp:
            fp.write('P2\n')
            fp.write('%d %d\n' % (n, m))
            fp.write('%d\n' % np.max(x[:, :, k]))
            for i in range(m):
                for j in range(n):
                    fp.write('%d ' % x[i, j, k])
                fp.write('\n')

    m, n, p = 2, 2, 2
    x = np.zeros((2, 2, 2))
    for h in range(256):
        x[ii, jj, kk] = [int(a)*255 for a in list('{0:08b}'.format(h))]
        file_dir = './data/cubes/%03d' % h
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        for k in range(2):
            file_name = '%s/%02d.pgm' % (file_dir, k + 1)
            with open(file_name, 'w') as fp:
                fp.write('P2\n')
                fp.write('%d %d\n' % (n, m))
                fp.write('%d\n' % np.max(x[:, :, k]))
                for i in range(m):
                    for j in range(n):
                        fp.write('%d ' % x[i, j, k])
                    fp.write('\n')


make_cubes()


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


def read_pgm(file_name):
    image = []
    with open(file_name, 'r') as fp:
        s = fp.readline().strip()
        assert s == 'P2'
        width, height = [int(x) for x in fp.readline().strip().split()]
        max_intensity = int(fp.readline().strip())

        max_int = float('-inf')
        for line in fp.readlines():
            row = [int(x) for x in line.strip().split()]
            assert len(row) == width
            image.append(row)
            max_int = max(max(row), max_int)

        assert len(image) == height
        assert max_int == max_intensity

    return image


def rot(a, x, y, z):
    # based on http://3dengine.org/Rotate_arb
    c = cos(a*np.pi/180)
    s = sin(a*np.pi/180)
    t = 1 - c
    return np.array([
        [t*x*x+c,    t*x*y-s*z,  t*x*z+s*y],
        [t*x*y+s*z,  t*y*y+c,    t*y*z-s*x],
        [t*x*z-s*y,  t*y*z+s*x,  t*z*z+c]
    ])


unit_cube = np.array([[2*int(b)-1 for b in list('{0:03b}'.format(a))] for a in range(8)])


def rotate(r, h):
    powers = np.array([2**i for i in range(7, -1, -1)])
    i = np.argsort(((unit_cube.dot(r)+1)/2).dot(powers[-3:]))
    return np.array([int(a) for a in list('{0:08b}'.format(h))]).dot(powers[i])


def invert(h):
    """
    invert: Inverts the bit pattern to check for the opposite case (in the 2d marching squares example).
    :param h:
    :return:
    """
    # TODO: Does this logic hold in 3d?
    powers = [2**i for i in range(7, -1, -1)]
    return np.array([1 - int(a) for a in list('{0:08b}'.format(h))]).dot(powers)


def create_mesh():
    global vertices, normals, triangles, points, image_height, image_width, image_depth

    image_directory = sys.argv[1]
    num_images = int(sys.argv[2])
    threshold = float(sys.argv[3])

    x = []
    for i in range(1, num_images+1):
        img = read_pgm('%s/%02d.pgm' % (image_directory, i))
        x.append(img)
    x = np.array(x)
    x = np.transpose(x, (1, 2, 0))

    image_width, image_height, image_depth = x.shape
    print(image_width, image_height, image_depth)

    x = (x > threshold).astype(int)

    points = []
    for i in range(image_width):
        for j in range(image_height):
            for k in range(image_depth):
                if x[i, j, k]:
                    print(i, j, k)
                    points.append(([i, j, k], [0, 1, 0]))
                else:
                    points.append(([i, j, k], [1, 0, 0]))
    points = np.array(points)

    # using rotations here: http://www.euclideanspace.com/maths/geometry/rotations/axisAngle/examples/index.htm
    sq33 = 3 ** (1/2) / 3
    sq22 = 2 ** (1/2) / 2
    rotations = [
        rot(0, 1, 0, 0),       # identity
        rot(90, 1, 0, 0),      # 90 deg about x
        rot(180, 1, 0, 0),     # 180 deg about x
        rot(-90, 1, 0, 0),     # 270 deg about x
        rot(90, 0, 1, 0),      # 90 deg about y
        rot(180, 0, 1, 0),     # 180 deg about y
        rot(-90, 0, 1, 0),     # 270 deg about y
        rot(90, 0, 0, 1),      # 90 deg about z
        rot(180, 0, 0, 1),     # 180 deg about z
        rot(-90, 0, 0, 1),     # 270 deg about z
        rot(120, sq33, sq33, sq33),     # 120 deg about ( 1, 1, 1) corner 7
        rot(-120, sq33, sq33, sq33),    # 120 deg about (-1,-1,-1) corner 0
        rot(120, sq33, sq33, -sq33),    # 120 deg about ( 1, 1,-1) corner 6
        rot(-120, sq33, sq33, -sq33),   # 120 deg about (-1,-1, 1) corner 1
        rot(120, sq33, -sq33, sq33),    # 120 deg about ( 1,-1, 1) corner 5
        rot(-120, sq33, -sq33, sq33),   # 120 deg about (-1, 1,-1) corner 2
        rot(120, sq33, -sq33, -sq33),   # 120 deg about ( 1,-1,-1) corner 4
        rot(-120, sq33, -sq33, -sq33),  # 120 deg about (-1, 1, 1) corner 3
        rot(180, sq22, sq22, 0),     # 180 deg about ( 1, 1, 0) edge 23
        rot(180, 0, sq22, sq22),     # 180 deg about ( 0, 1, 1) edge 02
        rot(180, -sq22, sq22, 0),    # 180 deg about (-1, 1, 0) edge 01
        rot(180, 0, sq22, -sq22),    # 180 deg about ( 0, 1,-1) edge 13
        rot(180, sq22, 0, sq22),     # 180 deg about ( 1, 0, 1) edge 26
        rot(180, -sq22, 0, sq22),    # 180 deg about (-1, 0, 1) edge 04
    ]
    rotation_repr = {
        0: 'identity',
        1: '90 deg about x ccw',
        2: '180 deg about x ccw',
        3: '270 deg about x ccw',
        4: '90 deg about y ccw',
        5: '180 deg about y ccw',
        6: '270 deg about y ccw',
        7: '90 deg about z ccw',
        8: '180 deg about z ccw',
        9: '270 deg about z ccw',
        10: '120 deg about ( 1, 1, 1) corner 7',
        11: '120 deg about (-1,-1,-1) corner 0',
        12: '120 deg about ( 1, 1,-1) corner 6',
        13: '120 deg about (-1,-1, 1) corner 1',
        14: '120 deg about ( 1,-1, 1) corner 5',
        15: '120 deg about (-1, 1,-1) corner 2',
        16: '120 deg about ( 1,-1,-1) corner 4',
        17: '120 deg about (-1, 1, 1) corner 3',
        18: '180 deg about ( 1, 1, 0) edge 23',
        19: '180 deg about ( 0, 1, 1) edge 02',
        20: '180 deg about (-1, 1, 0) edge 01',
        21: '180 deg about ( 0, 1,-1) edge 13',
        22: '180 deg about ( 1, 0, 1) edge 26',
        23: '180 deg about (-1, 0, 1) edge 04'
    }

    missed = 0
    # TODO: Fill in vertices and normals for each triangle here
    vertices = []
    norms = {
        'right': (1, 0, 0),
        'left': (-1, 0, 0),
        'top': (0, 1, 0),
        'bot': (0, -1, 0),
        'front': (0, 0, 1),
        'back': (0, 0, -1)
    }
    normals = []
    triangles = []

    """
         c --------- g
       /   |       / |
      /    |      /  |
     /     a ----/---e
    d ----/---- h    /
    |    /     |    /
    |   /      |   /
    |  /       |  /
    b -------- f /
    a: x=-1, y=-1, z=-1
    b: x=-1, y=-1, z=1
    c: x=-1, y=1, z=-1
    d: x=-1, y=1, z=1
    e: x=1, y=-1, z=-1
    f: x=1, y=-1, z=1
    g: x=1, y=1, z=-1
    h: x=1, y=1, z=1
    
    weight: 128, 64, 32, 16, 8, 4, 2, 1
    vertex:   h,  g,  f,  e, d, c, b, a
    access: x[i+1][j+1][k+1],...,x[i][j][k] 
    """
    # Code for marching cubes:
    '''
    URGENT: When coding the following, use the identity rotation as a base case, otherwise will be off by factor of r.
    '''
    lines = []
    for i in range(image_width - 1):
        for j in range(image_height - 1):
            for k in range(image_depth - 1):
                '''
                h0 = (a*1) + (b*2) + (c*4) + (d*8) + (e*16) + (f*32) + (g*64) + (h*128)
                '''
                # h0 = x[i][j][k] + (x[i][j][k+1] * 2) + (x[i][j+1][k] * 4) + (x[i+1][j][k+1] * 8) \
                #      + (x[i][j+1][k] * 16) + (x[i][j+1][k+1] * 32) + (x[i+1][j+1][k] * 64) + (x[i+1][j+1][k+1] * 128)
                h0 = x[i][j][k] + (x[i][j][k+1] * 2) + (x[i][j+1][k] * 4) + (x[i][j+1][k+1] * 8) \
                     + (x[i+1][j][k] * 16) + (x[i+1][j][k+1] * 32) + (x[i+1][j+1][k] * 64) + (x[i+1][j+1][k+1] * 128)
                # h0 = int(h0, 2)
                print('h0: %d' % h0)
                # Rotate cube 24 different ways to see if any cases match:
                for r_num, r in enumerate(rotations):
                    print('\tr: %s' % rotation_repr[r_num])
                    # h is the cube after applying a rotation:
                    h = rotate(r, h0)
                    print('\th: %d' % h)
                    # We also want to check the opposite contour case for efficiency:
                    ih = invert(h)
                    print('\tih: %d' % ih)
                    # Now check all of the 15 cases for a match:
                    if h == 0 or ih == 0:
                        pass
                        break
                    elif h == 1 or ih == 1:
                        # wikipedia case 1
                        # tested on: pmg 001, pgm 002, pgm 004, pgm 008, pgm 016, pgm 032, pgm 064, pgm 128
                        vertices.append([-1.0, 0.0, -1.0])
                        vertices.append([0.0, -1.0, -1.0])
                        vertices.append([-1.0, -1.0, 0.0])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        # TODO: Fix the norms.
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 3 or ih ==3:
                        # wikipedia case 2
                        # First triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([0, -1, 1])
                        vertices.append([-1, 0, 1])
                        # Second triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([0, -1, 1])
                        vertices.append([0, -1, -1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 6 or ih == 6:
                        # wikipedia case 3
                        # First triangle:
                        vertices.append([-1, 0, 1])
                        vertices.append([-1, -1, 0])
                        vertices.append([0, -1, 1])
                        # Second triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([-1, 1, 0])
                        vertices.append([0, 1, -1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 19 or ih == 19:
                        # Wikipedia case 4
                        # tested on: pgm 019, pgm 076
                        # First triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([-1, 0, 1])
                        vertices.append([1, 0, -1])
                        # Second triangle:
                        vertices.append([-1, 0, 1])
                        vertices.append([1, 0, -1])
                        vertices.append([1, -1, 0])
                        # Third triangle:
                        vertices.append([-1, 0, 1])
                        vertices.append([1, -1, 0])
                        vertices.append([0, -1, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 15 or ih == 15:
                        # wikipedia case 5
                        # First triangle:
                        vertices.append([0, -1, 1])
                        vertices.append([0, 1, 1])
                        vertices.append([0, -1, -1])
                        # Second triangle:
                        vertices.append([0, -1, -1])
                        vertices.append([0, 1, -1])
                        vertices.append([0, 1, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 147 or ih == 147:
                        # wikipedia case 6
                        # tested on: pgm 147
                        # First triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([-1, 0, 1])
                        vertices.append([1, 0, -1])
                        # Second triangle:
                        vertices.append([-1, 0, 1])
                        vertices.append([0, 1, 1])
                        vertices.append([1, 1, 0])
                        # Third triangle:
                        vertices.append([-1, 0, 1])
                        vertices.append([1, 1, 0])
                        vertices.append([1, 0, -1])
                        # Fourth triangle:
                        vertices.append([0, -1, 1])
                        vertices.append([1, -1, 0])
                        vertices.append([1, 0, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 105 or ih == 105:
                        # TODO: Resolve ih inverse error with case 7. Specifically, should the lines really be drawn in
                        #   the same place when the vertices are off? How to distinguish between rotation?
                        # wikipedia case 7
                        # tested on: pgm 105, pgm 150
                        # First Triangle:
                        vertices.append([-1, 1, 0])
                        vertices.append([-1, 0, -1])
                        vertices.append([0, 1, -1])
                        # vertices.append([-1, 0, -1])
                        # vertices.append([-1, -1, 0])
                        # vertices.append([0, -1, -1])
                        # Second Triangle:
                        vertices.append([1, 1, 0])
                        vertices.append([0, 1, 1])
                        vertices.append([1, 0, 1])
                        # Third Triangle:
                        vertices.append([0, -1, -1])
                        vertices.append([1, -1, 0])
                        vertices.append([1, 0, -1])
                        # vertices.append([-1, -1, 0])
                        # vertices.append([0, -1, -1])
                        # vertices.append([-1, 0, -1])
                        # Fourth Triangle:
                        vertices.append([-1, -1, 0])
                        vertices.append([-1, 0, 1])
                        vertices.append([0, -1, 1])
                        # vertices.append([1, -1, 0])
                        # vertices.append([0, -1, 1])
                        # vertices.append([1, 0, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 43 or ih == 43:
                        # Wikipedia case 8
                        # Tested on: pgm 43, pgm 232, pgm 212
                        # NOTE: PGM 232 may not be displaying correct with rotation; hard to tell without full controls.
                        # First Triangle:
                        vertices.append([0, -1, -1])
                        vertices.append([1, -1, 0])
                        vertices.append([-1, 0, -1])
                        # Second Triangle:
                        vertices.append([1, -1, 0])
                        vertices.append([-1, 0, -1])
                        vertices.append([-1, 1, 0])
                        # Third Triangle:
                        vertices.append([1, -1, 0])
                        vertices.append([-1, 1, 0])
                        vertices.append([1, 0, 1])
                        # Fourth Triangle:
                        vertices.append([1, 0, 1])
                        vertices.append([-1, 1, 0])
                        vertices.append([0, 1, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 46 or ih == 46:
                        # TODO: Fix case 9 under rotation with dr. parry's help.
                        # TODO: Actually this case is just badly broken in general. Re-do pgm tests.
                        # Wikipedia case 9
                        # Tested on: pgm 83, pgm 202, pgm 141, pgm 92,
                        # Case 15 tests: pgm 27,
                        # First Triangle:
                        vertices.append([1, 0, -1])
                        vertices.append([0, 1, -1])
                        vertices.append([-1, 1, 0])
                        # Second Triangle:
                        vertices.append([-1, 1, 0])
                        vertices.append([-1, -1, 0])
                        vertices.append([0, -1, 1])
                        # Third Triangle:
                        vertices.append([1, 0, -1])
                        vertices.append([1, 0, 1])
                        vertices.append([0, -1, 1])
                        # Fourth Triangle:
                        # Note: Comment this triangle out for ease of debugging
                        vertices.append([-1, 1, 0])
                        vertices.append([0, -1, 1])
                        vertices.append([1, 0, -1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 24 or ih == 24:
                        # Wikipedia case 10
                        # Tested on: pgm 24, pgm 129
                        # First Triangle:
                        vertices.append([-1, 1, 0])
                        vertices.append([-1, 0, 1])
                        vertices.append([0, 1, 1])
                        # Second Triangle:
                        vertices.append([0, -1, -1])
                        vertices.append([1, -1, 0])
                        vertices.append([1, 0, -1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    elif h == 44 or ih == 44:
                        # Wikipedia case 11 (coded on pgm 052)
                        # Tested on: pgm 44, pgm 52, pgm 098, pgm 131
                        # similar: pgm 067
                        # First Triangle:
                        vertices.append([1, 0, 1])
                        vertices.append([1, -1, 0])
                        vertices.append([0, -1, 1])
                        # Second Triangle:
                        vertices.append([0, 1, -1])
                        vertices.append([0, 1, 1])
                        vertices.append([-1, 0, -1])
                        # Third Triangle:
                        vertices.append([-1, 0, -1])
                        vertices.append([0, 1, 1])
                        vertices.append([-1, 0, 1])
                        vertices = np.array(vertices)
                        vertices = vertices.dot(r)
                        vertices = (vertices + 1) / 2
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        normals.append(norms['top'])
                        break
                    else:
                        print('Failed to match %d' % h0)
                        continue
                print('rotation', r)
                normals = np.array(normals)


def main():
    global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id
    create_mesh()
    # load the appropriate pgm

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

    # callbacks
    glutDisplayFunc(display)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)

    glutMainLoop()


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


if __name__ == '__main__':
    main()
