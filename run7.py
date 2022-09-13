import os
import json
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from combine_runs import combine_seeds, combine_subruns


CONFIG = {
            "days": 10000,
            "districts": ["City Of Tshwane Metro", "City Of Johannesburg Metro",
                "Ekurhuleni Metro", "Sedibeng", "West Rand"],
            "N": [3522325, 5538596, 3781377, 952102, 922640],
            "Asymptomatic_Cases": [0, 0, 0, 0, 0],
            "Mild_Cases": [0, 0, 0, 0, 0],
            "Hospitalised_Cases": [0, 0, 0, 0, 0],
            "ICU_Cases": [0, 0, 0, 0, 0],
            "Recovered": [0, 0, 0, 0, 0],
            "Deaths": [0, 0, 0, 0, 0],
            "t_incubation": [4, 1],
            "t_infective": [2.5, 3],
            "vulnerability": [[2.33, 23.2], [3.96, 20.2],[8.11, 21.1], [6.00, 16.7], [5.22, 21.6]],
            "vulnerability_start": 0.8,
            "vulnerability_range": 0.4,
            "R0": [40.5, 0.05],
            "upsilon": [[0.001978, 0.622763]] * 5,
            "rho": 0.75,
            "vacc_eff": 0.8071682187,
            "alpha_i23": [5,13],
            "gamma_i3": [[0.002282, 44.18015]] * 5,
            "alpha_i24": [1,12],
            "delta_i3": [[0.888672, 3.122973]] * 5,
            "gamma_i4": [[0.003537, 1.821067]] * 5,
            "delta_i4": [[161.1025, 0.153571]] * 5,
            "mobility": [[0.8115834442, 0.0796074971, 0.0804403798, 0.013268029, 0.01510065],
            [0.0289415763, 0.6365418334, 0.294823539, 0.0186296065, 0.0210634447],
            [0.0300457299, 0.2856778345, 0.6191559164, 0.0326120901, 0.032508429],
            [0.0090392005, 0.0284042544, 0.0485215909, 0.5427908922, 0.3712440619],
            [0.0101938026, 0.0298346009, 0.0457389387, 0.3696105686, 0.5446220892]]
}

#create file in the repo directory to save the results of the test run
MODEL_DIR = 'SENS_23'

#Use a number of seeds to perform a MC algorithm 
SEEDS = range(70001, 80000)

#Create a function that runs a SEIRDV model for each seed
def run(seed):
    config = deepcopy(CONFIG)
    config['seed'] = seed
    
    #The with statement save the rates you put in a dictionary to json
    with open(os.path.join(MODEL_DIR, 'config_start.json'), 'w') as writer:
        json.dump(config, writer, indent=2)
        writer.close()
    #set-up the multi-SEIRDV model- This calculates all 5 SEIRDV models and lets them interact with each other using the mobility matrix
    cmd = 'python3 sensitivity_analysis_main_multi.py'
    cmd += f" --config {os.path.join(MODEL_DIR, 'config_start.json')}"
    cmd += f" --config_out {os.path.join(MODEL_DIR, 'config_out_' + str(seed) + '.json')}"
    cmd += f" --data_out {os.path.join(MODEL_DIR, 'run_' + str(seed) + '.csv')}"

    os.system(cmd)

#execute
if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    for seed in SEEDS:
        run(seed)
