"""
radials.py
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
import matplotlib.lines

__author__ = "Chris Campell"
__created__ = "3/28/2018"


def main():
    upper_left = (-1, 1)
    upper_right = (1, 1)
    lower_left = (-1, -1)
    lower_right = (1, -1)
    # If we use this corner arrangement that is commented out we get a z instead of a square.
    # corners = np.array([upper_left, upper_right, lower_left, lower_right])
    corners = np.array([upper_left, upper_right, lower_right, lower_left])
    print(corners)
    print(corners.dot(rot(90)))
    # plot four points and the lines between them for box.
    # each point a different color
    plt.scatter(x=corners[:, 0], y=corners[:, 1], c=range(len(corners)))
    plt.plot(list(corners[:, 0]) + list(corners[:, 1]),
             list(corners[:, 1]) + list(corners[:, 0]), c='black')
    # plt.plot(corners[:, 0], corners[:, 1], c='k')
    plt.show()



def rot(a):
    # based on https://en.wikipedia.org/wiki/Rotation_matrix
    c = cos(a*np.pi/180)
    s = sin(a*np.pi/180)
    return np.array([
        [c, -s],
        [s, c]
    ])

if __name__ == '__main__':
    main()
