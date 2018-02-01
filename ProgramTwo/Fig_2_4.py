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
    X = np.delete(iris.values, 4, axis=1)

    # Resource: https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results
    # Resource: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html

    # Standardize:
    X_std = StandardScaler().fit_transform(X=X)
    # Covariance matrix:
    # cov = np.cov(X_std.T)
    # Eigenvectors and Eigenvalues via Eigendecomposition
    # ev, eig = np.linalg.eig(cov)
    # PCA:
    # pca = eig.dot(X_std.T)

    # Perform PCA:
    pca = decomposition.PCA(n_components=2)
    iris_pca = pca.fit_transform(X=X, y=None)
    # Multiply transformed data by -1 to revert mirror image:
    iris_pca[:,0] = iris_pca[:,0] * (-1)

    # iris_pca[:,1] = iris_pca[:,1] * (-1)

    # Plot Results:
    fig, ax = plt.subplots()
    # verts the distance between
    y_verts = []
    plt.scatter(x=iris_pca[:,0], y=iris_pca[:,1], marker=(4, 1, 90))
    plt.scatter(x=[x*-1 for x in X_std[:,0]], y=X_std[:,1], marker=lines.TICKUP, c='green')
    # plt.scatter(x=iris_pca[:,1], y=iris_pca[:,0], marker=(4, 1, 90))
    # Get the difference in position along the y-axis from PCA y-coord and original.
    # y_diff = [x-y for x,y in zip(iris_pca[:,1], iris.values[:,1])]
    y_diff = [y1-y2 for y1,y2 in zip(iris_pca[:,1], iris.values[:,1])]
    x_diff = [x-y for x,y in zip(iris_pca[:,0], iris.values[:,0])]
    # plt.errorbar(x=iris_pca[:,0], y=iris_pca[:,1], yerr=y_diff, c='green')
    plt.title('Principle Components of the Iris Dataset')
    plt.xlabel('Principle Component One')
    plt.ylabel('Principle Component Two')
    plt.show()

if __name__ == '__main__':
    main()