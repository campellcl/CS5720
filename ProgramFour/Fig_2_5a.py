"""
Fig_2_5a.py
Implementation of figure 2.5a from the course textbook.
Resources:
    https://plot.ly/python/parallel-coordinates-plot/#reference
"""

__author__ = "Chris Campell"
__version__ = "2/18/2018"

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# Plot.ly API AuthO:
plotly.tools.set_credentials_file(username='ccampell', api_key='utMKtuFvZHQE8N9RnRfP')

# Load data:
with open(sys.argv[1], 'r') as fp:
    iris = pd.read_csv(fp)

# Rename columns:
renamed_cols = list(iris.columns)
renamed_cols[0] = 'sepal length (cm)'
renamed_cols[1] = 'sepal width (cm)'
renamed_cols[2] = 'petal length (cm)'
renamed_cols[3] = 'petal width (cm)'
iris.columns = renamed_cols
iris['class'] = iris['class'].astype('category')
#print(iris.dtypes)
iris['class_cat'] = iris['class'].cat.codes
# iris.head()

data = [
    go.Parcoords(
        line=dict(color=iris['class_cat'],
                  colorscale=[[0, '#D7C16B'], [0.5, '#23D8C3'], [1, '#F3F10F']]),
        dimensions=list([
            dict(range=[min(iris['sepal length (cm)']), max(iris['sepal length (cm)'])],
                 label='Sepal Length (cm)', values=iris['sepal length (cm)']),
            dict(range=[min(iris['sepal width (cm)']), max(iris['sepal width (cm)'])],
                 label='Sepal Width (cm)', values=iris['sepal width (cm)']),
            dict(range=[min(iris['petal length (cm)']), max(iris['petal length (cm)'])],
                 label='Petal Length (cm)', values=iris['petal length (cm)']),
            dict(range=[0, max(iris['petal width (cm)'])],
                 label='Petal Width (cm)', values=iris['petal width (cm)'])
        ])
    )
]

layout = go.Layout(
    title='Figure 2.5a',
    plot_bgcolor='#E5E5E5',
    paper_bgcolor='#E5E5E5',
    xaxis=dict(
        fixedrange=True
    ),
    yaxis=dict(
        fixedrange=True
    )
)

fig = go.Figure(data=data, layout=layout)
plot = py.iplot(fig, filename='fig_2_5a', staticplot=True)
#print("Plot URL: https://plot.ly/~ccampell/4/figure-25a/")
print("Plot URL: %s" % plot.resource)
py.sign_in(username='ccampell', api_key='utMKtuFvZHQE8N9RnRfP')
py.image.save_as(fig, filename='fig_2_5a.png')
