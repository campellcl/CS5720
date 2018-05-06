import sys
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos


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


# return (i0, j0, i1, j1)
def top(i, j):
    return i, j + 0.5


def bottom(i, j):
    return i + 1, j + 0.5


def left(i, j):
    return i + 0.5, j


def right(i, j):
    return i + 0.5, j + 1


def lower_left(i, j):
    return [left(i, j), bottom(i, j)]


def lower_right(i, j):
    return [bottom(i, j), right(i, j)]


def upper_right(i, j):
    return [top(i, j), right(i, j)]


def upper_left(i, j):
    return [left(i, j), top(i, j)]


def horizontal(i, j):
    return [left(i, j), right(i, j)]


def vertical(i, j):
    return [top(i, j), bottom(i, j)]


code_to_line = {
    0: lambda i, j: [],
    1: lambda i, j: [lower_left(i, j)],
    2: lambda i, j: [lower_right(i, j)],
    3: lambda i, j: [horizontal(i, j)],
    4: lambda i, j: [upper_right(i, j)],
    5: lambda i, j: [upper_left(i, j), lower_right(i, j)],
    6: lambda i, j: [vertical(i, j)],
    7: lambda i, j: [upper_left(i, j)],
    8: lambda i, j: [upper_left(i, j)],
    9: lambda i, j: [vertical(i, j)],
    10: lambda i, j: [lower_left(i, j), upper_right(i, j)],
    11: lambda i, j: [upper_right(i, j)],
    12: lambda i, j: [horizontal(i, j)],
    13: lambda i, j: [lower_right(i, j)],
    14: lambda i, j: [lower_left(i, j)],
    15: lambda i, j: []
}


def rot(a):
    # based on https://en.wikipedia.org/wiki/Rotation_matrix
    c = cos(a*np.pi/180)
    s = sin(a*np.pi/180)
    return np.array([
        [c, -s],
        [s, c]
    ])


unit_square = np.array([[2*int(b)-1 for b in list('{0:02b}'.format(a))] for a in range(4)])


def h2im(h):
    return np.array([int(a) for a in list('{0:04b}'.format(h))])[[2, 3, 0, 1]].reshape((2, 2)).T


def rotate(r, h):
    powers = [2**i for i in range(3, -1, -1)]
    i = np.argsort(((unit_square.dot(r)+1)/2).dot(powers[-2:]))
    return np.array([int(a) for a in list('{0:04b}'.format(h))])[i].dot(powers)


def invert(h):
    powers = [2**i for i in range(3, -1, -1)]
    return np.array([1 - int(a) for a in list('{0:04b}'.format(h))]).dot(powers)


def gen_image(x):
    height = len(x)
    width = len(x[0])

    rotations = [
        rot(0),
        rot(90),
        rot(180),
        rot(-90)
    ]

    lines = []
    for i in range(height - 1):
        for j in range(width - 1):
            """
            v1 --- v3
            |      |
            |      |
            v0 --- v2

            v0: i=1, j=0, x = -1, y = -1
            v1: i=0, j=0, x = -1, y = +1
            v2: i=1, j=1, x = +1, y = -1
            v3: i=0, j=1, x = +1, y = +1

            weight: 8, 4, 2, 1
            vertex: 3, 2, 1, 0
            """
            h0 = x[i][j + 1] * 8 + x[i + 1][j + 1] * 4 + + x[i][j] * 2 + x[i + 1][j] * 1
            print('h0', h0)
            line_cases = []
            for r in rotations:
                h = rotate(r, h0)
                ih = invert(h)
                if h == 0 or ih == 0:
                    pass
                    break
                elif h == 1 or ih == 1:
                    line_cases = [np.array([(-1, 0), (0, -1)])]
                    break
                elif h == 5 or ih == 5:
                    line_cases = [np.array([(-1, 0), (1, 0)])]
                    break
                elif h == 6 or ih == 6:
                    line_cases = [np.array([(-1, 0), (0, 1)]), np.array([(0, -1), (1, 0)])]
                    break
            else:
                print('Failed to match %d' % h0)
                continue
            print('rotation', r)
            print('h', h)

            for line_case in line_cases:
                # invert the rotation
                line_rot = line_case.dot(r.T)
                # convert from (x, y) to (i, j)
                line_rot[:, 1] = -line_rot[:, 1]
                line_scale = (line_rot + 1) / 2
                line_scale[:, [0, 1]] = line_scale[:, [1, 0]]

                line_image = line_scale + np.array([[i, j]])
                new_lines = [line_image]
                lines.extend(new_lines)

    plt.clf()
    plt.imshow(x, cmap='gray', origin='upper')
    for line in lines:
        (y0, x0), (y1, x1) = line
        plt.plot([x0, x1], [y0, y1], 'r-')

    # plt.savefig(sys.argv[3])


def main():
    file_name = sys.argv[1]
    threshold = int(sys.argv[2])
    im = read_pgm(file_name)
    im = [[int(a > threshold) for a in row] for row in im]
    gen_image(im)
    plt.title('%s, threshold=%d' % (file_name, threshold))
    plt.colorbar()
    plt.show()


def test():
    for h in range(16):
        im = h2im(h)
        print(im)
        print('close figure window to proceed')
        gen_image(im)
        plt.title('h = %d' % h)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()
