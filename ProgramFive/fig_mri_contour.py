"""
FigMRIContour.py
Implementation of the marching squares algorithm for CS5720.
"""

import sys
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

__author__ = "Chris Campell"
__version__ = "2/26/2018"

def threshold(isovalue, pgm):
    """
    threshold: Applys a threshold to the 2D feild to make a binary image containing 1 where the data is above the
        provided isovalue, 0 where the data is below the provided value.
    :param isovalue: The threshold to apply to the image.
    :return pgm_thresholded: The provided PGM file with a 1 where the data is above the provided isovalue, and zeros
        elsewhere.
    """
    pgm_thresholded = np.zeros(shape=pgm.shape)
    for i, row in enumerate(pgm):
        for j, col in enumerate(row):
            if pgm[i, j] > isovalue:
                pgm_thresholded[i, j] = 1
    return pgm_thresholded

with open(sys.argv[1], 'r') as fp:
    pgm_str = fp.read()
    pgm_str = pgm_str.split('\n')
    dims = (int(pgm_str[1].split(' ')[0]), int(pgm_str[1].split(' ')[1]))
    max_value = int(pgm_str[2])
    pgm_str = pgm_str[3:]
    pgm_str = pgm_str[:-1]
    pgm = np.zeros(shape=dims)
    for i, string_row in enumerate(pgm_str):
        for j, string_col in enumerate(string_row.split(' ')):
            pgm[i,j] = int(string_col)
    # source: http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html
    contours = measure.find_contours(array=pgm, level=sys.argv[2])
    fig, ax = plt.subplots()
    cax = ax.imshow(pgm, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('%s, threshold=%s' %(sys.argv[1], sys.argv[2]))
    cbar = fig.colorbar(cax, ticks=np.arange(0,round(max_value, 1),10), orientation='vertical')
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
    plt.savefig(sys.argv[3])
