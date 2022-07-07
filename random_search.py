import os
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import json

districts = {"City Of Tshwane Metro": 3522325,
            "City Of Johannesburg Metro": 5538596,
            "Ekurhuleni Metro": 3781377,
            "Sedibeng": 952102,
            "West Rand": 922640}

def rate_run(predictions, data):
    config = predictions.replace('/run_', '/config_out_').replace('.csv', '.json')
    reader = open(config, 'r')
    config = json.load(reader)
    reader.close()
    predictions = pd.read_csv(predictions)

    pred_cases = []
    true_cases = []
    pred_hosp = []
    true_hosp = []
    for i, dist in enumerate(districts):
        n = districts[dist]
        mild = predictions[f'{dist}_Exposed'].values * n
        mild *= (1 - config['p1'][i]) / config['Incubation_Period'][i]
        pred_cases.append(mild)
        hosp = predictions[f'{dist}_Mild'].values * n
        hosp *= config['alpha_i2'][i]
        pred_hosp.append(hosp)

        true_cases.append(data[f'New Cases {dist}'].values)
        true_hosp.append(data[f'{dist} Hospitalised Cases'].values)

    pred_cases = sum(pred_cases)
    true_cases = sum(true_cases)
    pred_hosp = sum(pred_hosp)
    true_hosp = sum(true_hosp)

    pred = pred_cases + pred_hosp
    true = true_cases + true_hosp

    true_peak = true.argmax()
    pred_peak = pred.argmax()

    predictions = predictions.loc[(pred_peak - true_peak):].reset_index()
    predictions = predictions.loc[:len(data)]

    r2 = []
    for i, dist in enumerate(districts):
        n = districts[dist]
        mild = predictions[f'{dist}_Exposed'] * n
        mild *= (1 - config['p1'][i]) / config['Incubation_Period'][i]
        data[f'{dist}_Mild'] = mild
        hosp = predictions[f'{dist}_Mild'] * n
        hosp *= config['alpha_i2'][i]
        data[f'{dist}_Hospitalised'] = hosp

        true = data[f'New Cases {dist}'].values
        pred = data[f'{dist}_Mild'].values
        pred = pred[true > 10]
        true = true[true > 10]
        #r2.append(r2_score(true, pred))
        r2.append(((true - pred) ** 2).mean())
        true = data[f'{dist} Hospitalised Cases'].values
        pred = data[f'{dist}_Hospitalised'].values
        pred = pred[true > 10]
        true = true[true > 10]
        #r2.append(r2_score(true, pred))
        r2.append(((true - pred) ** 2).mean())

    return sum(r2) / len(r2)


if __name__ == '__main__':
    path = 'SENS_21'
    runs = [file for file in os.listdir(path) if 'run_' in file]
    seeds = [int(run.split('_', 1)[1].replace('.csv', '')) for run in runs]

    true_hosp = 'Data/wave1_pred_rep_hosp.csv'
    true_cases = 'Data/wave1_pred_rep_mild.csv'

    data_hosp = pd.read_csv(true_hosp)
    data_cases = pd.read_csv(true_cases)

    data = data_cases.join(data_hosp.set_index('Date'), on='Date')
    data = data.dropna().reset_index()

    r2 = []
    for seed in tqdm(seeds):
        predictions = f'{path}/run_{seed}.csv'
        r2.append(rate_run(predictions, data))
    r2 = np.array(r2)
    seeds = np.array(seeds)

    idx = r2.argsort()[::-1]
    idx = seeds[idx]

    writer = open('best_ids.json', 'w')
    json.dump(idx.tolist(), writer)
    writer.close()
