import numpy as np
from datetime import datetime
import pandas as pd
import generate_time_series as gts
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
import math
import data_preprocess as dp
from tabulate import tabulate
import helper_functions as hf
from statsmodels.tsa.arima_model import AR, ARIMA, ARMA
import matplotlib.pyplot as plt

class ts_forecast:
    def __init__(self, ts_dict):
        self.ts_dict = ts_dict

    def tS_forecast(self):
        with open('sdgs_time_series.txt', 'w') as sdgs:
            for i in self.ts_dict:
                for df in (self.ts_dict[i]):
                    if len(df) == 18:
                        v = df[df['Value'] == 0]
                        if (len(v) < 3):
                            #print(tabulate(df, headers='keys', tablefmt='psql'))
                            n = 4
                            # years_to_predict = range(2018, 2021, 1)

                            df['Year'] = df['Year'].astype(str).apply(lambda x: hf.obtain_date(x))
                            df.set_index('Year', inplace=True)
                            plt.plot(df)

                            rol_mean = pd.rolling_mean(df, window=10)
                            rol_std = pd.rolling_std(df, window=10)

                            # Plot rolling statistics:
                            orig = plt.plot(df, color='blue', label='Original')
                            mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
                            std = plt.plot(rol_std, color='black', label='Rolling Std')
                            plt.legend(loc='best')
                            plt.title('Rolling Mean & Standard Deviation')
                            plt.show(block=False)

                            # Perform Dickey-Fuller test:
                            # print
                            # 'Results of Dickey-Fuller Test:'
                            # dftest = adfuller(timeseries, autolag='AIC')
                            # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used',
                            #                                          'Number of Observations Used'])
                            # for key, value in dftest[4].items():
                            #     dfoutput['Critical Value (%s)' % key] = value
                            # print
                            # dfoutput

                            # d = df.values
                            #
                            # train, validate = d[:(len(df) - n)], d[(len(df) - n):]
                            # # uv_data = list(autoArDfr['Value'])
                            #
                            # validate_bench = []
                            # for i in validate:
                            #     for j in i:
                            #         validate_bench.append(j)

                            # #baseline set of models for time series forecasting
                            #models = {'AR': AR(train), 'ARMA': ARMA(train, order=(0, 1))}
                            # for i in (models):
                            #     mod = models[i].fit()
                            #     print('MODEL : {}'.format((i)))
                            #     print('Coefficients: %s' % mod.params)
                                #predictions = mod.predict(start=len(train), end=len(train) + len(validate)-1, dynamic=False)
                                # for j in range(len(predictions)):
                                #     print('Predicted - {:.4f} yet Actual Value is {:.4f}'.format(predictions[j], validate_bench[j]))
                                # print('RMSE is {}\n'.format(math.sqrt(mean_squared_error(validate_bench, predictions))))

                            ##baseline auto-arima
                            # model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
                            # a_arima = model.fit(train)
                            # arima_predict = a_arima.predict(n_periods=len(validate))
                            # forecast = a_arima.predict(n_periods=len(validate) + 15)  # forecasting the next 5 years
                            #
                            # print('\nMODEL: ARIMA')
                            # for j in range(len(arima_predict)):
                            #     print('Predicted - {:.4f} yet Actual Value is {:.4f}'.format(arima_predict[j], validate_bench[j]))
                            # auto_arima_df = pd.DataFrame(forecast, columns=['Prediction'])
                            # print('RMSE is {}'.format(math.sqrt(mean_squared_error(validate_bench, arima_predict))))
                            # print('MAE is ', mean_absolute_error(validate_bench, arima_predict))
                            # for x, y in zip(years_to_predict, forecast[-13:]):
                            #     print('Prediction for {} is {}'.format(x, y))

if __name__=='__main__':
    data_processed = dp.data_preprocessing('egypt.csv')
    data_reshaped = gts.generate_time_series(data_processed.reshaping())
    if data_reshaped:
        unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
        print('The number of Unique Series_description in Egypt: {}'.format(len(unique_sds_in_country)))
        data_reshaped.generate_combinations_per_series_des(unique_sds_in_country)
        # sub_sub_indicators = data_reshaped.sd_time_series_dict
        # for i in sub_sub_indicators:
        #     for j in sub_sub_indicators[i]:
        #         print(tabulate(j, headers='keys', tablefmt='psql'))

        f = ts_forecast(data_reshaped.sd_time_series_dict)
        f.tS_forecast()
        # for i in f.ts_dict:
        #     for j in (f.ts_dict[i]):
        #         v = j[j['Value'] == 0]
        #         if(len(v) < 3):
        #             print(tabulate(j, headers='keys', tablefmt='psql'))