import pandas as pd
import numpy as np
import os
import re
import json
import math
import pprint
from glob import glob
from pyexpat.errors import codes
from tabulate import tabulate
from collections import defaultdict

def hierarchical_enforcement(tax_file, country_file, ts_json):
    e = []
    df = pd.read_excel(tax_file)

    df['thresholds'] = df['thresholds'].str.lower()
    df.ffill(inplace=True)
    #df.to_csv('taxonomythresholdsfinal-pretty.csv')

    #get the unwanted meta data
    df_country = pd.read_csv(country_file, low_memory=False)
    un_wanted_meta_data = list(set(df_country['Units'])) + list(set(df_country['GeoAreaCode'])) + list(set(df_country['GeoAreaName'])) + list(set(df_country['[Disability status]']))

    un_wanted_meta_data = [str(i).strip() for i in un_wanted_meta_data]

    jf_file = open(ts_json, 'r')
    jf_file = json.load(jf_file)

    codes_of_interest_json = [extract_sereis_code_target_goal(i['Series_description'], un_wanted_meta_data)[0] for i in jf_file]
    codes_of_interest_taxonomy = [(search_sixth_underscore(i.strip(), '_', 6))[0] for i in df['SeriesCode']]

    df['SC'] = codes_of_interest_taxonomy
    d = [i for i in codes_of_interest_taxonomy if i not in codes_of_interest_json]

    #extracted_thresholds_taxonomy_of_interest

    #extracting the portion of interest from the large taxonomy for a particular country
    ex_thx_tax = pd.DataFrame()
    for v in set(codes_of_interest_json):
        r = df[df['SC'] == v]
        ex_thx_tax = pd.concat([ex_thx_tax, r], ignore_index=True)

    #print(tabulate(ex_thx_tax, headers='keys', tablefmt='psql'))


    for i in jf_file:
        prediction = i['FB_prophet'][-1]
        #-4 is the index of the original value at positidaon 2016, therefore we use it to track value at position 2015
        values = i['org_values'][:-3]

        #pick value in 2015, or if not there, check for the most recent non-null value
        for u in range(len(values) - 1, 0, -1):
            if not math.isnan(values[u]):
                value_2015 = values[u]
                break
            else:
                pass

        series_code, series_code_meta_data = extract_sereis_code_target_goal(i['Series_description'], un_wanted_meta_data)
        x = series_code+'_'+series_code_meta_data
        x = x.strip('_')
        for v in np.array(ex_thx_tax).tolist():
            goal, targ, th, ind, sd, se_c, sc = str(v[0]).strip(), str(v[1]).strip(), str(v[2]).strip(), v[4].strip(), v[5].strip(), v[6].strip(), v[7].strip()
            s_c, s_c_meta_data = search_sixth_underscore(se_c, '_', 6)
            y = s_c+'_'+s_c_meta_data
            y = y.strip('_')
            if x == y:
                if th.__contains__('='):
                    sign, threshold = search_sign_threshold(str(th))
                    if sign == '<' and th.__contains__('%'):
                        expected_2030 = (value_2015 * (100 - threshold)) / 100
                        if (prediction <= expected_2030):
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '>' and th.__contains__('%'):
                        expected_2030 = (value_2015 * (100 + threshold)) / 100
                        if (prediction >= expected_2030):
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '<' and not th.__contains__('%'):
                        if (prediction <= value_2015):
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'F'))
                    elif sign == '>' and not th.__contains__('%'):
                        if (prediction >= expected_2030):
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'T'))
                        else:
                            e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'F'))
                else:
                    e.append((goal, targ, ind, sd, s_c, s_c_meta_data, value_2015, prediction, 'Currently Unknown Threshold'))
            # else:
            #     print('This series code was not found {}'.format(x))

    country_benchmark = pd.DataFrame(e, columns=['Goal', 'Target', 'Indicator', 'Series Description', 'Series Code', 'Meta data', 'Initial Value', 'Prediction', 'Result'])
    country_benchmark_dend = country_benchmark[['Goal', 'Target', 'Indicator','Series Code', 'Meta data', 'Result']]
    #country_benchmark_dend.to_csv('egypt_bench_mark_one.csv')
    #print(tabulate(country_benchmark_dend, headers='keys', tablefmt='psql'))
    codes_of_interest_json = [extract_sereis_code_target_goal(i['Series_description'], un_wanted_meta_data)[0] for i in
                              jf_file]

    e = [i for i in set(codes_of_interest_json) if i not in set(country_benchmark_dend['Series Code'])]
    print('Not in the Threshold taxonomy but found in Json {}'.format(e))
    return country_benchmark_dend, e

#alldetails after the sixth underscore is meta data which makes a series description unique,
# e.g. DC_TOF_HLTHL_3_3.b_3.b.2_, has got series-code followed by goal, then target then indicator, and they're the one's you can use to search for details when looking through other files
def search_sixth_underscore(stri, char, n):
    o = []
    substring, meta_data = '', ''
    stri = str(stri).strip()
    if len(re.findall(char, stri, re.IGNORECASE)) < n: #this retains the string as it is if the series description has no meta data
        substring = stri
    else:
        r = [i for i in re.finditer(char, stri, re.IGNORECASE)]
        substring = stri[:(r[5].start())]
        meta_data = stri[(r[5].start()):]
        meta_data = meta_data.replace('__', '_')
        meta_data = sorted(meta_data.split('_'), key=lambda x:len(x), reverse=True)
        meta_data = '_'.join(i for i in meta_data).rstrip('_')

        for i in range(len(stri.strip())):
            if stri[i] == '_':
                if (len(o) <= n):
                    o.append(i)
                else:
                    t = i
                    break
            else:
                pass
    return substring.strip('_'), meta_data

def extract_sereis_code_target_goal(str_input, unwanted):
    str_input = [i for i in str_input.split('+^') if i not in unwanted]
    y = []
    u = []
    for i in str_input:
        i = str(i).strip()
        if re.search(r'_', i):
            y.append(i)
        elif i.__contains__('.'):
            if len(re.findall(r'\.', i)) == 2:
                j = i.split('.')
                try:
                    y.append(j[0])
                    y.append(j[0]+'.'+j[1])
                    y.append(j[0]+'.'+j[1]+'.'+j[2])
                except Exception as e:
                    print(i, e)
            else:
               pass
        else:
            try:
                int(i)
            except:
                u.append(i)

    y = sorted(y, key=lambda x:len(x))
    u = sorted(u, key=lambda x:len(x), reverse=True)
    e = y.pop()
    y.insert(0, e)
    y = '_'.join(i for i in y).strip('_')
    u = '_'.join(i for i in u).rstrip('_')

    return y, u


def search_sign_threshold(stri):
    match_threshold_string = re.match('(\<|\>)=(\d*\.*\d*)(%)?', stri, re.IGNORECASE)
    if match_threshold_string:
        sign = match_threshold_string.group(1)
        actual_threshold_value = match_threshold_string.group(2)
        return sign, float(actual_threshold_value)

if __name__== '__main__':
    #look through current director for all sub-directories assuming every sub-directory belongs to a country
    current_country_dirs = list(dir for dir in os.listdir('../SDGS/') if os.path.isdir(dir))
    #escape sub-directories the likes of '.git', '__pycache__' and many others
    current_country_dirs = [i for i in current_country_dirs if not re.search(r'\.|\_', i, re.IGNORECASE)]
    #open each sub-directory

    for i, country in enumerate(current_country_dirs):
        print(country)
        #pick only json files, because they're all you want to encode
        json_fname = glob(os.path.join(os.path.abspath(country), '*.json'))
        if json_fname:
            json_file_list = [i for i in json_fname if i.__contains__('series')]
            json_file = json_file_list[0]
            json_dir_file = os.path.dirname(json_file)

            country_dict = []
            with open(os.path.join(json_dir_file, 'errors.txt'), 'w') as err, open(os.path.join(json_dir_file, '{}-d3.json'.format(country)), 'w') as stratify:
                # try:
                dendogram_file, e = hierarchical_enforcement('taxonomythesholdsv3.xlsx', 'allcountries.csv', json_file)

                if len(e) > 0:
                    err.write('{} : Not in the Threshold taxonomy but found in Json {}'.format(country, e))

                dendogram = dendogram_file.sort_values(by=['Goal'])
                dendogram['m'] = dendogram_file['Series Code'].astype(str) + '__' + dendogram_file['Meta data'].astype(str) + '__'+ dendogram_file['Result'].astype(str)
                dendogram = dendogram.drop(['Series Code', 'Meta data', 'Result'], axis=1)
                #dendogram_file.to_csv(os.path.join(json_dir_file, '{}.csv'.format(country))

                countryObj = {}
                countryObj['name'] = country
                countryObj['parent'] = 'World'

                countryObj['children'] = []

                for g in set(dendogram['Goal']):
                    goalObj = {}
                    goal = int(float(g))
                    goal_target = list(set([t for t in dendogram['Target'] if str(t).split('.')[0] == str(goal)]))
                    goalObj['name'] = goal
                    goalObj['parent'] = country

                    goalObj['children'] = []

                    for g_t in goal_target:
                        #     goalObj['children'].append(k)
                        targetObj = {}
                        goal_target_indicator = list(set([i for i in dendogram['Indicator'] if '.'.join(str(i).split('.', 2)[:2]) == str(g_t)]))
                        targetObj['name'] = g_t
                        targetObj['parent'] = goal

                        targetObj['children'] = []

                        for g_t_i in goal_target_indicator:
                            #targetObj['children'].append(g_t_i)
                            indObj = {}
                            indObj['name'] = g_t_i
                            indObj['parent'] = g_t

                            indObj['children'] = []
                            goal_target_indicator_meta = list(set([s for s in dendogram['m'] if s.__contains__(g_t_i)]))
                            if goal_target_indicator_meta:
                                #print(goal, g_t, g_t_i, goal_target_indicator_meta)
                                for b in goal_target_indicator_meta:
                                    w = b.split('__')
                                    w1 = ['_'.join(i for i in w[0].split('_')[:3])]
                                    if len(w) > 2:
                                        b = '__'.join(i for i in w1+w[-2:]).replace('Currently Unknown Threshold', 'CUT')
                                        print(b)
                                    else:
                                        b = ''.join([i for i in w1+w[-1]]).replace('Currently Unknown Threshold', 'CUT')
                                        print(b)
                                    metObj = {}
                                    metObj['name'] = b
                                    metObj['parent'] = g_t_i

                                    indObj['children'].append(metObj)

                                targetObj['children'].append(indObj)
                        goalObj['children'].append(targetObj)
                    countryObj['children'].append(goalObj)
                country_dict.append(countryObj)
                #pprint.pprint(country_dict)
                json.dump(country_dict, stratify, indent=3)
                # except Exception as e:
                #     print(e)
                #     err.write('{}\n{}\n'.format(country, e))

        break

