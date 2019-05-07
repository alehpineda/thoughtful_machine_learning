import random
import sys

import pandas as pd
import numpy as np

from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

class Regression(object):
    """
    Regresion kNN
    """

    def __init__(self):
        self.k = 5
        self.metric = np.mean
        self.kdtree = None
        self.houses = None
        self.values = None


    def set_data(self, houses, values):
        """
        Setea valores de houses y values
        :param houses: pandas.DataFrame with houses parameters
        :param values: pandas.Series with houses values
        """

        self.houses = houses
        self.values = values
        self.kdtree = KDTree(self.houses)


    def regress(self, query_point):
        """
        Calcula el valor predicho para una casa con ciertos parametros 
        :param query_point: pandas.Series con los parametros de house
        :return: valor house
        """
        _, indexes = self.kdtree.query(query_point, self.k)
        value = self.metric(self.values.iloc[indexes])
        if np.isnan(value):
            raise Exception('Valor inesperado')
        else:
            return value



