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
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from fbprophet import Prophet
import json
import warnings
import sys

warnings.filterwarnings("ignore", category=Warning)

class ts_forecast:
    def __init__(self, ts_dict):
        self.ts_dict = ts_dict

    def tS_forecast(self):
        n = 4
        t = []
        json_dict = {}
        with open('sdgs_time_series_fb.json', 'w') as sdgs:
            with open('./PLOTS/combinations.txt', 'a') as comb_file:
                for sd in self.ts_dict:
                    df = self.ts_dict[sd]
                    if len(df) == 18:
                        v = df[df['Value'] == 0]
                        #ensure you have atleast 4 values are not null values
                        if (len(v) <= 13):
                            years_to_predict = range(2000, 2031, 1)
                            #change the date column to a date type
                            df['Year'] = df['Year'].astype(str).apply(lambda x: hf.obtain_date(x))
                            #df.set_index('Year', inplace=True)

                            #UNDO ONLY WHEN USING FACEBOOKPROPHET
                            df.columns = ['ds','y2']
                            y = [np.NaN if i==0 else i for i in df.y2]
                            df.drop('y2', inplace=True, axis=1)
                            df['y'] = y
                            print(tabulate(df, headers='keys', tablefmt='psql'))

                #             #sd_p_value = exmine_the_stationarity(df, sd, comb_file, send_p_values=False)
                #
                #             # UNDO ONLY WHEN TESTING THE NULL HYPOTHESIS OF THE STATIONARITY
                #             # if sd_p_value > 0.05:
                #             #     find_d(sd, df)
                #             #     validity_of_determined_p_d_q(sd, df)
                #
                            train, validate = df[:(len(df) - n)], df[(len(df) - n):]
                            actual = validate.y.tolist()
                            #FORECAST
                            try:
                #                 #ARIMA
                #                 # arima = ARIMA(train, order=(0, 1, 1))
                #                 # model_arima = arima.fit(disp=0)
                #                 # fc, se, conf = model_arima.forecast(8, alpha=0.05)  # 95% conf
                #                 # #Make as pandas series
                #                 # fc_series = pd.Series(fc, index=years_to_predict)
                #                 # lower_series = pd.Series(conf[:, 0], index=years_to_predict)
                #                 # upper_series = pd.Series(conf[:, 1], index=years_to_predict)
                #                 # predictions = fc.tolist()[:4]
                #                 # json_dict['Series_description'] = sd
                #                 # json_dict['Predicted_period'] = list(years_to_predict)
                #                 # json_dict['Arima'] = fc.tolist()
                #                 # json_dict['RMSE'] = prediction_accuracies(predictions, actual)
                #                 #
                #                 #
                #                 # #UNDO TO VISUALIZE HOW PREDICTIONS ARE TRENDING vs EXPECTATIONS
                #                 # # plt.figure(figsize=(12, 5), dpi=100)
                #                 # # plt.plot(train, label='training')
                #                 # # plt.plot(validate, label='actual')
                #                 # # plt.plot(fc_series, label='forecast')
                #                 # # plt.fill_between(lower_series.index, lower_series, upper_series,
                #                 # #                  color='k', alpha=.15)
                #                 # # plt.title('Forecast vs Actuals')
                #                 # # plt.legend(loc='upper left', fontsize=8)
                #                 # # plt.show()
                #                 #
                #                 # #BASELINE AUTO-ARIMA
                #                 # model_autoarima = auto_arima(train,
                #                 #                              trace=True,
                #                 #                              test='adf',
                #                 #                              max_p=3, max_q=3,
                #                 #                              m=1,
                #                 #                              d = None,
                #                 #                              seasonal=False,
                #                 #                              error_action='ignore',
                #                 #                              suppress_warnings=True,
                #                 #                              stepwise=True)
                #                 # a_arima = model_autoarima.fit(train)
                #                 # arima_forecast = a_arima.predict(n_periods=len(years_to_predict))  # forecasting the next 4 years
                #                 # predictions = arima_forecast.tolist()[:4]
                #                 # json_dict['Series_description'] = sd
                #                 # json_dict['Predicted_period'] = list(years_to_predict)
                #                 # json_dict['Auto_ARIMA'] = arima_forecast.tolist()
                #                 # json_dict['RMSE'] = prediction_accuracies(predictions, actual)
                #                 #
                #                 #
                                #FACEBOOK PROHPET
                                fb = Prophet(weekly_seasonality=True)
                                fb_model = fb.fit(train)
                                future = fb_model.make_future_dataframe(periods=len(years_to_predict))
                                fb_predict = fb.predict(future)
                                predictions = list(fb_predict['yhat'])[:18][-4:]
                                actual = validate.y.tolist()
                                json_dict['Series_description'] = sd
                                json_dict['Predicted_period'] = list(years_to_predict)[-13:]
                                json_dict['FB_prophet'] = list(fb_predict['yhat'])[-13:]
                                r_mse, mape = prediction_accuracies(predictions[:4], actual)
                                json_dict['RMSE'] = r_mse
                                json_dict['MAPE'] = mape
                                json_dict['org_values'] = list(df['y'])
                #
                #                 #BASELINE ARMA
                #                 # arma = ARMA(train, order=(0,1))
                #                 # arma_model = arma.fit(disp=0)
                #                 # #print('Coefficients: %s' % arma_model.params)
                #                 # arma_predict = arma_model.predict(start=len(train), end=len(train) + 7, dynamic=False)
                #                 # predictions = arma_predict.tolist()
                #                 # json_dict['Series_description'] = sd
                #                 # json_dict['Predicted_period'] = list(years_to_predict)
                #                 # json_dict['ARMA'] = predictions
                #                 # r_mse, mape = prediction_accuracies(predictions[:4], actual)
                #                 # json_dict['RMSE'] = r_mse
                #                 # json_dict['MAPE (%)'] = mape
                #
                                t.append(json_dict)
                                json_dict = {}

                            except Exception as e:
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                print(exc_type)
                                print(exc_obj)
                                print(exc_tb.tb_lineno)

                json.dump(t, sdgs, indent=2, sort_keys=True)
                sdgs.close()



#take a close look at rolling stats
def examine_rolling_metrics(df):
    rol_mean = df.rolling(2).mean()
    rol_std = df.rolling(2).std()

    # Plot rolling statistics:
    orig = plt.plot(df, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

#examine the stationarity of the time series
def exmine_the_stationarity(df, sd, file, send_p_values=False):
    result = adfuller(df.Value.dropna())
    if send_p_values == True:
        comb_file = open(file, 'a')
        try:
            comb_file.write('\n{}'.format(sd))
            comb_file.write('\nADF Statistic: {:.4f}'.format(result[0]))
            comb_file.write('\np-value: {:.4f}'.format(result[1]))
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])

        except Exception as e:
            print(sd)
            print(e)
        comb_file.close()
    return result[1]

#p,d,q exmination
def find_d(sd, df):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.Value);
    axes[0, 0].set_title('Original')
    plt.title('{}'.format(sd))
    plot_acf(df.Value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.Value.diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.Value.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.Value.diff().diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.Value.diff().diff().dropna(), ax=axes[2, 1])
    plt.show()

def find_p(sd, df):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.Value);
    axes[0, 0].set_title('Original')
    plt.title('{}'.format(sd))
    plot_pacf(df.Value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.Value.diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_pacf(df.Value.diff().dropna(), ax=axes[1, 1])

    plt.show()

def validity_of_determined_p_d_q(sd, df):
    try:
        model = ARIMA(df, order=(0, 1, 1))
        model_fit = model.fit(disp=0)
        model_fit.plot_predict(dynamic=False)
        plt.xticks(list(df.index))
        plt.show()

    except Exception as e:
        print(sd)
        print(e)

def prediction_accuracies(prediction, actual):
    difference = np.subtract(prediction, actual)
    mean_abs_err = np.nanmean(np.abs(difference))
    rmse = np.nanmean(difference**2)**.5
    return rmse, np.mean(difference)


if __name__=='__main__':
    data_processed = dp.data_preprocessing('egypt.csv')
    data_reshaped = gts.generate_time_series(data_processed.reshaping())
    if data_reshaped:
        unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
        print('The number of Unique Series_description in Egypt: {}'.format(len(unique_sds_in_country)))
        data_reshaped.generate_combinations_per_series_des(unique_sds_in_country)
        # sub_sub_indicators = data_reshaped.sd_time_series_dict
        # for i in sub_sub_indicators:
        #     print(i)
        #     for j in sub_sub_indicators[i]:
        #         print(j)
        #     break
                #print(tabulate(j, headers='keys', tablefmt='psql'))
        #output a json containing
        f = ts_forecast(data_reshaped.sd_time_series_dict)
        f.tS_forecast()
        # for i in f.ts_dict:
        #     for j in (f.ts_dict[i]):
        #         v = j[j['Value'] == 0]
        #         if(len(v) < 3):
        #             print(tabulate(j, headers='keys', tablefmt='psql'))


        #baseline set of models for time series forecasting
                        # models = {'AR': AR(train), 'ARMA': ARMA(train, order=(0, 1))}
                        # for i in (models):
                        #     mod = models[i].fit()
                        #     print('MODEL : {}'.format((i)))
                        #     print('Coefficients: %s' % mod.params)
                        #     predictions = mod.predict(start=len(train), end=len(train) + len(validate)-1, dynamic=False)
                        #     for j in range(len(predictions)):
                        #         print('Predicted - {:.4f} yet Actual Value is {:.4f}'.format(predictions[j], validate_bench[j]))
                        #     print('RMSE is {}\n'.format(math.sqrt(mean_squared_error(validate_bench, predictions))))


                        # print('\nMODEL: ARIMA')
                        # for j in range(len(arima_predict)):
                        #     print('Predicted - {:.4f} yet Actual Value is {:.4f}'.format(arima_predict[j], validate_bench[j]))
                        # auto_arima_df = pd.DataFrame(forecast, columns=['Prediction'])
                        # print('RMSE is {}'.format(math.sqrt(mean_squared_error(validate_bench, arima_predict))))
                        # print('MAE is ', mean_absolute_error(validate_bench, arima_predict))
                        # for x, y in zip(years_to_predict, forecast[-13:]):
                        #     print('Prediction for {} is {}'.format(x, y))