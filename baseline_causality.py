from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
from pprint import pprint
from scipy.stats import ttest_rel, shapiro, mannwhitneyu

pd.options.display.max_columns = 30

class baseline_causality(object):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        #self.data.set_index('Year', inplace=True)

    def examine_correlation(self):
        x = list(self.data.index)
        y = self.data.iloc[:,0:]
        self.data.plot.line()
        # plt.line(x, y)
        # plt.show()

    def grange_caudality(self):
        #print(tabulate(self.data, headers='keys', tablefmt='psql'))
        grangercausalitytests(self.data.iloc[:,0:4], maxlag=5, verbose=True)

    def lasso_model(self, x, y):
        b_model = Lasso(normalize=True, max_iter=2)
        b_model.fit(x, y)
        pred = b_model.predict(x)
        r_squared = r2_score(y, pred)
        return pred, r_squared

    def normality_test(self):
        normality_test = self.data.apply(lambda x: shapiro(x)[1] < 0.05)

    def t_test(self, alpha):
        org = self.data
        dup = org.copy()
        print(tabulate(org, headers='keys', tablefmt='psql'))
        print(list(org.columns))
        print(org.shape)
        interaction = np.zeros((org.shape[1], org.shape[1]))
        for i in range(org.shape[1]):
            for j in range(dup.shape[1]):
                if j < (dup.shape[1]-1):
                    t_stat, p_value = mannwhitneyu(org.iloc[:,i], dup.iloc[:,j+1])
                if p_value > alpha: #there's an interaction between the two variables
                    interaction[i][j+1] = p_value
                else:
                    pass
            dup = dup.iloc[:,1:]
        np.set_printoptions(precision=4)
        for i in interaction:
            print(i)
        interaction_frame = pd.DataFrame(interaction, index=list(org.columns), columns=list(org.columns))
        return interaction_frame

def rescaling(x, scaling_method=None):
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise MethodError(scaling_method, 'Sorry, use either one of these minmax, standard or robust').message
    scaled = scaler.fit_transform(x)
    return scaled


class MethodError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
    def __str__(self):
        return self.message

def scatter_plotting(x, y, pred, labels=[]):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.plot(x, pred, color='green')
    plt.show()

if __name__ == '__main__':
    country = 'Egypt'
    files = os.listdir('COMBINATIONS/{}'.format(country))
    files = [i for i in files if i.__contains__('csv')]
    for i in files:
        i_path = os.path.abspath('COMBINATIONS/{}/{}'.format(country,i))
        d = baseline_causality(i_path)
        df = d.data
        year = df.iloc[:,0]
        t_series = df.iloc[:,1:]
        scaled = rescaling(t_series, scaling_method='minmax')
        scaled_df = pd.DataFrame(scaled, columns=list(t_series.columns))
        scaled_df = pd.concat([year, scaled_df], axis=1)
        scaled_df.set_index('Year', inplace=True)
        scaled_df_len = scaled_df.shape[1]
        for t in range(scaled_df_len):
            if t < scaled_df_len-1:
                for pair_t in range(t+1, scaled_df_len):
                    x = scaled_df.iloc[:, t].values.reshape(-1, 1)
                    y = scaled_df.iloc[:,pair_t].values.reshape(-1, 1)
                    y_pred, r_squared = d.lasso_model(x, y)
                    scatter_plotting(x, y, y_pred, labels=['x', 'y'])
        print(scaled_df_len)



