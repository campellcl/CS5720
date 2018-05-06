"""
MRI3D.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = "Chris Campell"

# Put logic for pgm file read:
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
np.save('slice14.npy', pgm)
# data = np.load('slice14.npy')
num_rows, num_cols = (dims[0], dims[1])
pgm = pgm / pgm.max()
i, j = np.meshgrid(range(num_cols), range(num_rows))
# pgm.shape == i.shape
# Here we reshape the matrix into a vector.
i = i.reshape((-1,))
j = j.reshape((-1,))
c = pgm.reshape((-1,))
plt.scatter(j, i, c=c, cmap='gray')
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(j, i, 0, c=c)
plt.savefig(sys.argv[2])
