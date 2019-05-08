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


class RegresionTest(object):
    """
    Lee los datos, calcula y grafica el error
    de la regresion kNN 
    """

    def __init__(self):
        self.houses = None
        self.values = None

    
    def load_csv_file(self, csv_file, limit = None):
        """
        Carga el archivo CSV con datos de las casas.
        :param csv_file: Nombre del archivo CSV
        :param limit: Limite del numero de filas
        """

        houses = pd.read_csv(csv_file, nrows=limit)
        self.values = houses['AppraisedValue']
        houses = houses.drop('AppraisedValue', 1)
        houses = (houses - houses.mean()) / (houses.max() - houses.min())
        self.houses = houses
        self.houses = self.houses[['lat', 'long', 'SqFtLot']]


    def tests(self, folds):
        """
        Calcular los errores promedio absolutos para los test
        :param folds: Numero de veces que los datos se repartieron
        :return: lista de los valores de los errores
        """

        holdout = 1 / float(folds)
        errors = []
        for _ in range(folds):
            values_regress, values_actual = self.test_regression(holdout)
            errors.append(mean_absolute_error(values_actual, values_regress))

        return errors

    
    def test_regression(self, holdout):
        """
        Calcula la regresion para datos fuera de la muestra
        :param holdout: Parte de los datos a probar [0,1]
        :return: tuple(y_regression, values_actual)
        """

        test_rows = random.sample(self.houses.index.tolist(), int(round(len(self.houses) * holdout)))
        train_rows = set(range(len(self.houses))) - set(test_rows)
        df_test = self.houses.ix[test_rows]
        df_train = self.houses.drop(test_rows)

        train_values = self.values.ix[train_rows]
        regression = Regression()
        regression.set_data(houses=df_train, values=train_values)

        values_regr = []
        values_actual = []

        for idx, row in df_test.iterrows():
            values_regr.append(regression.regress(row))
            values_actual.append(self.values[idx])

        return values_regr, values_actual

    def plot_error_rates(self):
        """
        Plot MAE vs #folds
        """
        folds_range = range(2,11)
        errors_df = pd.DataFrame({'max':0, 'min':0}, index=folds_range)
        
        for folds in folds_range:
            errors = self.tests(folds)
            errors_df['max'][folds] = max(errors)
            errors_df['min'][folds] = min(errors)
        
        errors_df.plot(title='Error promedio absoluto de kNN con diferente rango de folds')
        plt.xlabel('#folds_range')
        plt.ylabel('MAE')
        plt.show()


import os

here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'king_county_data_geocoded.csv')


def main():
    regression_test = RegresionTest()
    regression_test.load_csv_file(filename)
    regression_test.plot_error_rates()

if __name__ == "__main__":
    main()

