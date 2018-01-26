"""
VisualizationOne.py
Implementation of Programming Assignment One for CS5720.
"""

__author__ = "Chris Campell"
__version__ = "1/25/2018"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm


# Load file:
with open('cars04.csv', 'r') as fp:
    data = pd.read_csv(fp, header=0)

# Convert to dataframe:
df_cars = pd.DataFrame(data=data)

# Remove extraneous columns:
df_cars = df_cars[['Vehicle Name', 'HP', 'City MPG']]

# Remove records with an unknown HP or City MPG:
df = df_cars.replace(r'/[*]/g', np.nan, regex=True)
# TODO: now that * has been replaced with np.nan; drop any rows where HP or MPG is np.nan!!!
df = df.dropna(axis=0, how='')
# Filter by Toyota vehicles:
toyota_only = df_cars[df_cars['Vehicle Name'].str.contains('Toyota')]
toyota_hp_vs_mpg = toyota_only[['Vehicle Name', 'HP', 'City MPG']]

# Create the scatter plot:
# Reference URL: https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
# https://stackoverflow.com/questions/4143502/how-to-do-a-scatter-plot-with-empty-circles-in-python
# http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%207%20-%20Cleaning%20up%20messy%20data.ipynb
x = df_cars['HP']
y = df_cars['City MPG']
fig, ax = plt.subplots()
# plt.scatter(x,y,marker='s',)
fig.colors = ['red', 'green', 'blue']
# y_min = int(toyota_hp_vs_mpg['City MPG'].min(0))
# y_max = int(toyota_hp_vs_mpg['City MPG'].max(0))
# x_min = int(toyota_hp_vs_mpg['HP'].min(0))
# x_max = int(toyota_hp_vs_mpg['HP'].max(0))
# plt.axis([x_min, x_max, y_min, y_max])
plt.scatter(x, y, marker='s', facecolors='None', edgecolor='black', linewidths=0.5)
# ax.scatter(x, y, marker='s', c='bue', facecolors='None')
# TODO: Okay to draw on top of existing points? Or do i want to filter first?
ax.scatter(toyota_only['HP'], toyota_only['City MPG'], marker='s', c='green')
plt.axis(y=np.arange(10,60,5),x=np.arange(73,500,42.7))
# plt.axis(y=np.arange(10,60,5))
# plt.yticks(np.arange(10.0,65.0,5.0))
plt.xticks(np.arange(73,542.7,42.7))
ax.legend()
plt.xlabel('HP')
plt.ylabel('City MPG')
plt.show()
