"""
FigMRIContour.py
Implementation of the marching squares algorithm for CS5720.
"""

import sys
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib import collections as mc

__author__ = "Chris Campell"
__version__ = "2/26/2018"

def build_lookup_table(num_bits=4):
    lookup_table = {}
    for i in range(2**num_bits):
        lookup_table['{0:b}'.format(i).zfill(num_bits)] = i
    return lookup_table

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

def contour(pgm_thresholded, patch=(2,2)):
    """
    contour: Forms contour cells based on the provided dimensionality and pgm.
    :param pgm: A Portable Gray Map (PGM) file.
    :param patch: The size of the contour cells.
    :return contours: The provided PGM converted to contour cells.
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
                for n, (x, y) in enumerate(contour_line_segments):
                    contour_segment.append((x+j, y+i))
                contours.append(contour_segment)
        contour_cells.append(contour_cell_row)
    contour_cells = np.array(contour_cells)

    # Give every cell a number based on which corners are true/false:
    # for i, contour_cell_row in enumerate(contour_cells):
    #     for j, contour_cell in enumerate(contour_cell_row):
    #         # Build the lookup-table binary index:
    #         cell_bin_index = ''
    #         for k, (x, y) in enumerate(contour_cell):
    #             # (x, y) is the indices of the contour cell in the original PGM.
    #             # Append to the lookup-table binary index:
    #             cell_bin_index = cell_bin_index + str(int(pgm_thresholded[x, y]))
    #         # Give every contour cell a number based on which corners are true/false (via lookup-table):
    #         # contour_pgm[i, j] = lookup_table[cell_bin_index]
    #         # TODO: This may return a series of up to 4 (x,y) tuples. What do?
    #         contour_line_segments = lookup_table[cell_bin_index]
    #         if contour_line_segments is not None:
    #             for (x, y) in contour_line_segments:
    #                 contours.append([x+j, y+i])
    # Go every every cell and replace its case number with the appropriate line plot:
    return contours



pgm = None
# This is set in the definition of the algorithm, and is not a magic number: (4 vertex in a square)
num_bits = 4

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

# lookup_table = build_lookup_table(num_bits=4)
lookup_table = {
    '0000': None,
    '0001': [(-1, 0), (0, -1)],
    '0010': [(0, -1), (1, 0)],
    '0011': [(-1, 0), (1, 0)],
    '0100': [(0, 1), (1, 0)],
    '0101': [(-1, 0), (0, 1), (0, -1), (1, 0)],
    '0110': [(0, 1), (0, -1)],
    '0111': [(-1, 0), (0, 1)],
    '1000': [(-1, 0), (0, 1)],
    '1001': [(0, 1), (0, -1)],
    '1010': [(-1, 0), (0, -1), (0, 1), (1, 0)],
    '1011': [(0, 1), (1, 0)],
    '1100': [(-1, 0), (1, 0)],
    '1101': [(0, -1), (1, 0)],
    '1110': [(-1, 0), (0, -1)],
    '1111': None
}
pgm_thresholded = threshold(pgm=pgm, isovalue=int(sys.argv[2]))
contours = contour(pgm_thresholded)

# source: http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html
# contours = measure.find_contours(array=pgm, level=sys.argv[2])
fig, ax = plt.subplots()
cax = ax.imshow(pgm, interpolation='nearest', cmap=plt.cm.gray)
# plt.xlim((-1, 1))
# plt.xticks(np.arange(-1, 1.5, .5))
# plt.ylim((-1, 1))
# plt.yticks(np.arange(-1, 1.5, .5))
plt.title('%s, threshold=%s' % (sys.argv[1], sys.argv[2]))
# plt.xticks(np.arange(0, len(pgm_thresholded), .5))
cbar = fig.colorbar(cax, ticks=np.arange(0, round(max_value, 1), 10), orientation='vertical')
print('Contour Lines: %s' % contours)
contour_lines_inverted = []
for contour_line in contours:
    contour_line_inv = []
    for x, y in contour_line:
        contour_line_inv.append((x, y*-1))
    contour_lines_inverted.append(contour_line_inv)

ax.set_xlim((-.5, 1.5))
ax.set_ylim((1.5, -.5))
lc = mc.LineCollection(contour_lines_inverted, linewidths=2, color='red', linestyles='solid')
ax.add_collection(lc)
# ax.autoscale()
# ax.margins(0.1)
# for n, contour in enumerate(contours):
#     plt.plot([x for (x, y) in contour], [y for (x, y) in contour], color='red', linewidth=1)
    # for x, y in contour:
    #     plt.plot(x, y, linewidth=1, color='red', linestyle='-')

plt.savefig(sys.argv[3])
