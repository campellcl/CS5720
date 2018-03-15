"""
FigMRIContour.py
Implementation of the marching squares algorithm for CS5720. Takes as input three command line arguments passed to
this script:
1. The name (and directory) of the .pgm file to read as input.
2. The value for which the .pgm file is to be thresholded.
3. The name (and/or directory) of the .png file to be saved as output.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

__author__ = "Chris Campell"
__version__ = "3/15/2018"


def threshold(isovalue, pgm):
    """
    threshold: Applys a threshold to the 2D field to make a binary image containing 1 where the data is above the
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


def contour(pgm_thresholded, patch=(2,2)):
    """
    contour: Forms contour cells based on the provided dimensionality and input thesholded PGM.
    :param pgm_thresholded: A Portable Gray Map (PGM) file that has been thresholded by the user-specified threshold.
    :param patch: The size of the contour cells.
    :return contours: A list of lists containing the start and end points for each contour line segment.
    """
    contours = []
    # First step is to look up the contour lines and put them into the cells.
    contour_cells = []
    for i in range(len(pgm_thresholded) - 1):
        contour_cell_row = []
        for j in range(len(pgm_thresholded[i]) - 1):
            contour_cell = [(i, j), (i, j+1), (i+1, j+1), (i+1, j)]
            contour_cell_row.append(contour_cell)
            # Create a binary index based on the values in the cell:
            cell_bin_index = ''.join(str(int(pgm_thresholded[x, y])) for x, y in contour_cell)
            # Perform a lookup using the binary index:
            contour_line_segments = lookup_table[cell_bin_index]
            contour_segment = []
            if contour_line_segments is not None:
                # If there are two lines present...
                if len(contour_line_segments) == 4:
                    for n, (x, y) in enumerate(contour_line_segments):
                        contour_segment.append((x+j, y+i))
                        if n == 1:
                            contours.append(contour_segment)
                            contour_segment = []
                        elif n == 3:
                            contours.append(contour_segment)
                else:
                    for n, (x, y) in enumerate(contour_line_segments):
                        contour_segment.append((x+j, y+i))
                    contours.append(contour_segment)
        contour_cells.append(contour_cell_row)
    return contours

pgm = None
# This is set in the definition of the algorithm, and is not a magic number: (there are 4 vertex in a square)
num_bits = 4

# Read the PGM:
with open(sys.argv[1], 'r') as fp:
    pgm_str = fp.read()
    pgm_str = pgm_str.split('\n')
    dims = (int(pgm_str[1].split(' ')[0]), int(pgm_str[1].split(' ')[1]))
    max_value = int(pgm_str[2])
    pgm_str = pgm_str[3:]
    pgm_str = pgm_str[:-1]
    pgm = np.zeros(shape=dims)
    for i, string_row in enumerate(pgm_str):
        for j, string_col in enumerate(string_row.split(' ')[:-1]):
            pgm[i,j] = int(string_col)

lookup_table = {
    '0000': None,
    '0001': [(0, 0.5), (0.5, 1.0)],
    '0010': [(0.5, 1), (1.0, 0.5)],
    '0011': [(0, 0.5), (1.0, 0.5)],
    '0100': [(0.5, 0), (1.0, 0.5)],
    '0101': [(0, 0.5), (0.5, 0), (0.5, 1.0), (1.0, 0.5)],
    '0110': [(0.5, 1.0), (0.5, 0)],
    '0111': [(0, 0.5), (0.5, 0)],
    '1000': [(0, 0.5), (0.5, 0)],
    '1001': [(0.5, 1.0), (0.5, 0)],
    '1010': [(0, 0.5), (0.5, 1.0), (0.5, 0), (1.0, 0.5)],
    '1011': [(0.5, 0), (1.0, 0.5)],
    '1100': [(0, 0.5), (1.0, 0.5)],
    '1101': [(0.5, 1.0), (1.0, 0.5)],
    '1110': [(0, 0.5), (0.5, 1.0)],
    '1111': None
}
pgm_thresholded = threshold(pgm=pgm, isovalue=int(sys.argv[2]))
contours = contour(pgm_thresholded)
# source: http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html
fig, ax = plt.subplots()
cax = ax.imshow(pgm, interpolation='nearest', cmap=plt.cm.gray)
plt.title('%s, threshold=%s' % (sys.argv[1], sys.argv[2]))
# Create the color bar:
cbar = fig.colorbar(cax, ticks=np.arange(0, round(max_value, 1), 10), orientation='vertical')
# Create a collection of line segments and plot the contour lines on the axis:
lc = mc.LineCollection(contours, linewidths=2, color='red', linestyles='solid')
ax.add_collection(lc)
plt.savefig(sys.argv[3])
