"""
Fig_2_4.py
Implementation of Programming Assignment Two for CS 5720.
"""

__author__ = "Chris Campell"
__version__ = "1/31/2018"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines


def main():
    # Load file:
    with open('iris.csv', 'r') as fp:
        iris = pd.read_csv(fp, header=0)
    # Search for NANs:
    print(iris.describe())
    print(iris.head())
    print(iris.info())

    # Remove Y (target):
    Y = iris['class']
    X = np.delete(iris.values, 4, axis=1)

    # Resource: https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results
    # Resource: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html

    # Standardize:
    X_std = StandardScaler().fit_transform(X=X)

    # Add back target labels for vis.
    # Perform PCA:
    pca = decomposition.PCA(n_components=4)
    sklearn_pca = pca.fit_transform(X=iris.values[:, 0:4], y=None)
    # Multiply transformed data by -1 to mirror image:
    sklearn_pca[:,0] = sklearn_pca[:,0] * (-1)
    # sklearn_pca[:,1] = sklearn_pca[:,1] * (-1)

    # Plot Results:
    fig, ax = plt.subplots()
    # plt.scatter(x=sklearn_pca[:,0], y=sklearn_pca[:,1], marker=(4, 1, 90))
    plt.scatter(x=sklearn_pca[0:50,0], y=sklearn_pca[0:50,1], marker='+', color='blue', label='Iris-setosa')
    plt.scatter(x=sklearn_pca[50:100,0], y=sklearn_pca[50:100,1], marker='+', color='green', label='Iris-versicolor')
    plt.scatter(x=sklearn_pca[100:,0], y=sklearn_pca[100:,1], marker='+', color='red', label='Iris-virginica')
    plt.title('Principle Components of the Iris Dataset')
    plt.xlabel('Principle Component One')
    plt.ylabel('Principle Component Two')
    plt.legend()
    plt.savefig('Fig_2_4.png')
    plt.show()

if __name__ == '__main__':
    main()
