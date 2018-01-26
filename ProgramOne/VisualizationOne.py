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

# Filter by Toyota vehicles:
toyota_only = df_cars[df_cars['Vehicle Name'].str.contains('Toyota')]
toyota_hp_vs_mpg = toyota_only[['Vehicle Name', 'HP', 'City MPG']]

# Create the scatter plot:
# Reference URL: https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
x = toyota_hp_vs_mpg['HP']
y = toyota_hp_vs_mpg['City MPG']
fig, ax = plt.subplots()
# plt.scatter(x,y,marker='s',)
fig.colors = ['red', 'green', 'blue']
# y_min = int(toyota_hp_vs_mpg['City MPG'].min(0))
# y_max = int(toyota_hp_vs_mpg['City MPG'].max(0))
# x_min = int(toyota_hp_vs_mpg['HP'].min(0))
# x_max = int(toyota_hp_vs_mpg['HP'].max(0))
# plt.axis([x_min, x_max, 0, 15])
ax.scatter(x,y)
plt.axis(y=np.arange(10,60,5),x=np.arange(73,500,42.7))
# plt.axis(y=np.arange(10,60,5))
# plt.yticks(np.arange(10.0,65.0,5.0))
plt.xticks(np.arange(73,542.7,42.7))
ax.legend()
plt.xlabel('HP')
plt.ylabel('City MPG')
plt.show()
