"""
Fig_2_5a.py
Implementation of figure 2.5a from the course textbook.
Resources:
    http://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html#parallel-coordinates
    https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    https://plot.ly/python/parallel-coordinates-plot/
"""

__author__ = "Chris Campell"
__version__ = "2/18/2018"

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

with open(sys.argv[1], 'r') as fp:
    iris = pd.read_csv(fp)

iris_data = iris.iloc[1:]
plt.figure()

parallel_coordinates(iris, 'class', color=['g','b','c'])
plt.show()
'''
# Find the max and min of each attribute for lines:
max(iris['sepal length'])
min(iris['sepal length'])
'''
