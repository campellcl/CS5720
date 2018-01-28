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

# Is there missing data?
df_cars.__str__().__contains__('*')

# Remove extraneous columns:
# Notice that 'Vehicle Name' is included because Figure 1.47 is only Toyotas
df_cars = df_cars[['Vehicle Name', 'HP', 'City MPG', 'Len', 'Width', 'Weight']]

# Remove records with an unknown HP, City MPG, Len, or Width:
df_cars = df_cars.replace(r'[*]', np.nan, regex=True)
df_cars = df_cars.dropna(axis=0, how='any')

# Add in column with vehicle area:
df_cars['Area'] = [int(l)*int(w) for l,w in zip(df_cars['Len'], df_cars['Width'])]

# Ensure all nan's have been dropped from 'HP':
# df = df[np.isfinite(df['HP'])]

# Filter by Toyota vehicles:
toyota_only = df_cars[df_cars['Vehicle Name'].str.contains('Toyota')]
toyota_hp_vs_mpg = toyota_only[['Vehicle Name', 'HP', 'City MPG', 'Area', 'Weight']]

# Create the scatter plot:
# Reference URL: https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
# https://stackoverflow.com/questions/4143502/how-to-do-a-scatter-plot-with-empty-circles-in-python
# http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%207%20-%20Cleaning%20up%20messy%20data.ipynb
x = df_cars['HP']
y = df_cars['City MPG']
fig, ax = plt.subplots()
fig.colors = ['red', 'green', 'blue']
# y_min = int(toyota_hp_vs_mpg['City MPG'].min(0))
# y_max = int(toyota_hp_vs_mpg['City MPG'].max(0))
# x_min = int(toyota_hp_vs_mpg['HP'].min(0))
# x_max = int(toyota_hp_vs_mpg['HP'].max(0))

# Let the size of the marker represent the weight of the vehicle:
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
plt.scatter(x, y, marker='s', s=df_cars['Weight'], facecolors='None', edgecolor='black', linewidths=0.5)


# ax.scatter(x, y, marker='s', c='bue', facecolors='None')
ax.scatter(toyota_only['HP'], toyota_only['City MPG'], marker='s', c='green')
plt.axis(y=np.arange(10, 60, 5), x=np.arange(73, 500, 42.7))
# plt.axis(y=np.arange(10,60,5))
# plt.yticks(np.arange(10.0,65.0,5.0))
plt.xticks(np.arange(73, 542.7, 42.7))
ax.legend()
plt.xlabel('Horse Power')
plt.ylabel('City Miles-Per-Gallon')
plt.title('')
plt.show()
