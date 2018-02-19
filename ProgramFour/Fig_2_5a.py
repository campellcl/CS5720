"""
Fig_2_5a.py
Implementation of figure 2.5a from the course textbook.
Resources:
    http://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html#parallel-coordinates
    https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    https://plot.ly/python/parallel-coordinates-plot/
    http://benalexkeen.com/parallel-coordinates-in-matplotlib/
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

plotly.tools.set_credentials_file(username='ccampell', api_key='utMKtuFvZHQE8N9RnRfP')

with open(sys.argv[1], 'r') as fp:
    iris = pd.read_csv(fp)

# TODO: Modify each individual y-axis to have different scale depending on max and min of attribute.

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
                 constraintrange=[4, 8],
                 label='Sepal Length (cm)', values=iris['sepal length (cm)']),
            dict(range=[min(iris['sepal width (cm)']), max(iris['sepal width (cm)'])],
                 label='Sepal Width', values=iris['sepal width (cm)']),
            dict(range=[min(iris['petal length (cm)']), max(iris['petal length (cm)'])],
                 label='Petal Length', values=iris['petal length (cm)']),
            dict(range=[min(iris['petal width (cm)']), max(iris['petal width (cm)'])],
                 label='Petal Width', values=iris['petal width (cm)'])
        ])
    )
]

layout = go.Layout(
    plot_bgcolor='#E5E5E5',
    paper_bgcolor='#E5E5E5',
    xaxis=dict(
        fixedrange=True
    ),
    yaxis=dict(
        fixedrange=True,
        showline=False,
        visible=False
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='parcoords-basic', staticplot=True)

#plt.figure()
# parallel_coordinates(iris, 'class', color=['g','b','c'])
# plt.savefig(sys.argv[2])
