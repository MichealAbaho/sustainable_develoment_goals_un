from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import Lasso
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
        self.data.set_index('Year', inplace=True)

    def examine_correlation(self):
        x = list(self.data.index)
        y = self.data.iloc[:,0:]
        self.data.plot.line()
        # plt.line(x, y)
        # plt.show()

    def grange_caudality(self):
        #print(tabulate(self.data, headers='keys', tablefmt='psql'))
        grangercausalitytests(self.data.iloc[:,0:4], maxlag=5, verbose=True)

    def lasso_model(self):
        b_model = Lasso()
        b_model.fit(self.data.iloc[:,8:14], self.data.iloc[:,7])
        plt.scatter(list(self.data['Value.6']), list(self.data['Value.13']))
        plt.show()
        print(b_model.coef_)

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


file = 'COMBINATIONS/analysis_file_3.2.csv'
d = baseline_causality(file)
#d.examine_correlation()
#d.lasso_model()
print(tabulate(d.t_test(0.05), headers='keys', tablefmt='psql'))
