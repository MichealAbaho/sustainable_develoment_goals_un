import pandas as pd
import numpy as np
import os
import re
import json
import math

from tabulate import tabulate

def hierarchical_enforcement(tax_file, ts_json):
    e = []
    df = pd.read_excel(tax_file)
    df['thresholds'] = df['thresholds'].str.lower()
    df.ffill(inplace=True)
    #df.to_csv('taxonomythresholdsfinal-pretty.csv')

    jf_file = open(ts_json, 'r')
    jf_file = json.load(jf_file)
    print(len(jf_file))

    codes_of_interest_json = [(search_sixth_underscore(i['Series_description'].strip(), '_', 6))[0] for i in jf_file]
    codes_of_interest_taxonomy = [(search_sixth_underscore(i.strip(), '_', 6))[0] for i in df['SeriesCode']]

    df['SC'] = codes_of_interest_taxonomy
    d = [i for i in codes_of_interest_taxonomy if i not in codes_of_interest_json]
    print('Number of series description not accounted for in Egypt:{}'.format(len(set(d))))
    print('Number of series description accounted for in Egypt:{}'.format(len(set(codes_of_interest_json))))
    #extracted_thresholds_taxonomy_of_interest

    ex_thx_tax = pd.DataFrame()
    for v in set(codes_of_interest_json):
        r = df[df['SC'] == v]
        ex_thx_tax = pd.concat([ex_thx_tax, r], ignore_index=True)

    #print(tabulate(ex_thx_tax, headers='keys', tablefmt='psql'))
    u = 0
    # print(tabulate(ex_thx_tax, headers='keys', tablefmt='psql'))
    for i in jf_file:
        prediction = i['FB_prophet'][-1]

        values = i['org_values'][:-2]
        #pick value in 2015, or if not there, check for the most recent non-null value
        for u in range(len(values) - 1, 0, -1):
            if not math.isnan(values[u]):
                value_2015 = values[u]
                break
            else:
                pass

        series_code, series_code_meta_data = search_sixth_underscore(i['Series_description'].strip(), '_', 6)
        series_code_meta_data = '_'.join(i for i in sorted([i for i in series_code_meta_data.lstrip('_').split('_') if i not in ['818.0', 'Egypt']], key=lambda x:len(x), reverse=True))
        #print(series_code)
        #for th, sc, goal, sd in zip(ex_thx_tax['thresholds'], ex_thx_tax['SeriesCode'], ex_thx_tax['Goal'], ex_thx_tax['SeriesDescription']):
        for v in np.array(ex_thx_tax).tolist():
            goal, targ, th, ind, sd, sc = str(v[0]).strip(), str(v[1]).strip(), str(v[2]).strip(), v[4].strip(), v[5].strip(), v[6]
            s_c, s_c_meta_data = search_sixth_underscore(sc, '_', 6)
            s_c_meta_data = '_'.join(i for i in sorted(s_c_meta_data.lstrip('_').split('_'), key=lambda x:len(x), reverse=True))

            if series_code+series_code_meta_data == s_c+s_c_meta_data:
                if th.__contains__('='):
                    sign, threshold = search_sign_threshold(str(th))
                    if sign == '<' and th.__contains__('%'):
                        expected_2030 = (value_2015 * (100 - threshold)) / 100
                        if (prediction <= expected_2030):
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '>' and th.__contains__('%'):
                        expected_2030 = (value_2015 * (100 + threshold)) / 100
                        if (prediction >= expected_2030):
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '<' and not th.__contains__('%'):
                        if (prediction <= value_2015):
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '>' and not th.__contains__('%'):
                        if (prediction >= expected_2030):
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'F'))
                else:
                    e.append((goal, targ, ind, sd, s_c+s_c_meta_data, value_2015, prediction, 'Currently Unknown Threshold'))
            else:
                pass
                #print('This series code was not found {}'.format(series_code))

    country_benchmark = pd.DataFrame(e, columns=['Goal', 'Target', 'Indicator', 'Series Description', 'Series Code', 'Initial Value', 'Prediction', 'Result'])
    #country_benchmark_one_frame = country_benchmark_one_frame.drop_duplicates(keep='first', subset=['Series Code'])
    country_benchmark_dend = country_benchmark[['Goal', 'Target', 'Indicator','Series Code','Result']]
    #country_benchmark_dend.to_csv('egypt_bench_mark_one.csv')
    # print(country_benchmark_one_frame.shape[0])

    return country_benchmark_dend

def search_sixth_underscore(stri, char, n):
    o = []
    substring, meta_data = '', ''
    stri = str(stri).strip()
    if len(re.findall(char, stri, re.IGNORECASE)) < n:
        substring = stri
    else:
        r = [i for i in re.finditer(char, stri, re.IGNORECASE)]
        substring = stri[:(r[5].start())]
        meta_data = stri[(r[5].start()):]
        for i in range(len(stri.strip())):
            if stri[i] == '_':
                if (len(o) <= n):
                    o.append(i)
                else:
                    t = i
                    break
            else:
                pass
    return substring, meta_data

def search_sign_threshold(stri):
    match_threshold_string = re.match('(\<|\>)=(\d*\.*\d*)(%)?', stri, re.IGNORECASE)
    if match_threshold_string:
        sign = match_threshold_string.group(1)
        actual_threshold_value = match_threshold_string.group(2)
        return sign, float(actual_threshold_value)

if __name__== '__main__':
    dendogram_file = hierarchical_enforcement('taxonomy thresholds final.xlsx', 'encoded_TS_fb.json')
    dendogram_file['Goal'] = dendogram_file['Goal'].astype(float)
    dendogram_file = dendogram_file.sort_values(by=['Goal'])
    #dendogram_file = dendogram_file[dendogram_file['Goal'] == 3]

    dendogram_file = dendogram_file.groupby(['Goal', 'Target', 'Indicator', 'Series Code', 'Indicator'])['Indicator'].count()
    #dendogram_file.to_json('x.json', orient='records')
    print(dendogram_file)
    #print(tabulate(dendogram_file, headers='keys', tablefmt='psql'))