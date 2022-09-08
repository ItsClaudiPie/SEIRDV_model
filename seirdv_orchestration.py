import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from default_config import CONFIG
from combine_runs import combine_seeds, combine_subruns

#create file in the repo directory to save the results of the test run
# MODEL_DIR = 'wave_1_0'
# MODEL_DIR = 'wave_2_0'
MODEL_DIR = 'wave_3_0'
# MODEL_DIR = 'wave_4_0'
# MODEL_DIR = 'wave_5_0'

#Use a number of seeds to perform a MC algorithm
n_seeds = 50
SEEDS = [i + 1 for i in range(n_seeds)]

#Create a function that runs a SEIRDV model for each seed
def run(seed):
    config = deepcopy(CONFIG)
    config['seed'] = seed
   
    # Wave 1

    config['days'] = 10000
    # immune_percentage = 0
    # config['Vaccinated'] = [immune_percentage * n for n in config['N']]
    # config['Vaccinated'] = [0, 0, 0, 0, 0]
    # config['upsilon'] = [0, 0, 0, 0, 0]
    # config["alpha_i24"] = [0.002, 0.001724137931034483, 0.0017857142857142859, 0.0014705882352941176, 0.001282051282051282]
    # config["alpha_i23"] = [0.002631578947368421, 0.0014285714285714286, 0.0029411764705882353, 0.0029411764705882353, 0.002272727272727273]
    # config["gamma_i3"] = [0.05645849085422254, 0.05059842484263016, 0.05641435245414137, 0.06750861079219289, 0.07250426453807762]
    # config["gamma_i4"] = [0.05412956016264819, 0.03598792261818881, 0.05682107718991267, 0.06997124469396139, 0.045738030687482126]
    # config["delta_i3"] = [0.035294089749680076, 0.025617013984876785, 0.03493207931373253, 0.054351487928130264, 0.0429594891135747]
    # config["delta_i4"] = [0.0414777878513146, 0.04288552603679941, 0.03355315111797479, 0.05303689814504788, 0.08517215950989847]
    # config["R0_lockdown_scale"] = 0.6
    config["vulnerability_start"] = 0.8
    config["vulnerability_range"] = 0.4
    

    # Wave 2

    #config['days'] = 250
    # immune_percentage = 0.3
    # config['Vaccinated'] = [immune_percentage * n for n in config['N']]
    # config['vacc_eff'] = 0.94
    # config['gamma_i3'] = [[0.07717, 1.295859], [0.077362, 1.271625], [0.078912, 1.296978],
    #                       [0.079364, 1.231145], [0.085019, 1.052753]]
    # config['gamma_i4'] = [[0.013134, 1.33916], [0.003642, 1.30373], [0.002868, 1.25178],
    #                       [0.002551, 3.336442], [3.336442, 0.204242]]
    # config['delta_i3'] = [[0.048709, 1.288133], [0.050393, 1.277347], [0.046518, 1.298361],
    #                       [0.001747, 44.39242], [0.019775, 3.878854]]
    # config['delta_i4'] = [[0.096414, 1.505558], [1.05E-01, 1.508105], [0.105575, 1.507514],
    #                       [0.000011, 14104.86], [0.038591, 4.192026]]
    # config['alpha_i24'] = [0.0025, 0.0019230769230769232, 0.002380952380952381, 0.0015625, 0.0020833333333333333]
    # config['alpha_i23'] = [0.0025, 0.0020833333333333333, 0.0025, 0.001724137931034483, 0.003125]
    #config['gamma_i3'] = [0.07563809626489125, 0.06001240570233004, 0.0669012732646232, 0.06624661376936403, 0.07574975448086897]
    #config['gamma_i4'] = [0.06991030493031666, 0.04561001299982771, 0.05212859529431979, 0.03567596762303757, 0.029472436277415533]
    #config['delta_i3'] = [0.03263849189684432, 0.02431795365969884, 0.04152554651396641, 0.04049768584652306, 0.04555432776054165]
    #config['delta_i4'] = [0.03895541209057694, 0.041014294231206, 0.07023290675401826, 0.07755581668625147, 0.11898982894128524]
    # config["Exposed"] = [10, 10, 4, 3,3]
    # config["Asymptomatic_Cases"] = [10, 10, 10, 3, 3]
    # config["Mild_Cases"] = [3, 4, 3, 3, 0]
    # config["Hospitalised_Cases"] = [0, 0, 0, 0, 0]
    # config["ICU_Cases"] = [0, 0, 0, 0, 0]
    # config["Recovered"] = [0, 0, 0, 0, 0]
    # config["Deaths"] = [0, 0, 0, 0, 0]
    # config["R0_lockdown_scale"] = 0.8

    # Wave 3

    config['gamma_i3'] = [[0.001797, 61.71666], [0.077362, 1.271625], [0.078912, 1.296978],
                          [0.079364, 1.231145], [0.085019, 1.052753]]
    config['gamma_i4'] = [[0.000835, 10.25527], [0.003642, 1.30373], [0.002868, 1.25178],
                          [0.003732, 1.37886], [0.00476, 1.279336]]
    config['delta_i3'] = [[0.063682, 0.891219], [0.050393, 1.277347], [0.046518, 1.298361],
                          [0.045222, 1.438175], [0.049606, 1.476454]]
    config['delta_i4'] = [[0.000139, 1134.626], [1.05E-01, 1.508105], [0.105575, 1.507514],
                          [0.104489, 1.508293], [0.10385, 1.508474]]

    #config['days'] = 250
    immune_percentage = 0.4
    config['Vaccinated'] = [immune_percentage * n for n in config['N']]
    config['vacc_eff'] = 0.67
    config['alpha_i24'] = [0.0029411764705882353, 0.0016666666666666668, 0.002380952380952381, 0.001388888888888889, 0.002173913043478261]
    config['alpha_i23'] = [0.003125, 0.0016666666666666668, 0.0038461538461538464, 0.0017857142857142859, 0.0014705882352941176]
    #config['gamma_i3'] = [0.10035415908761203, 0.0681861907804632, 0.07558082532581459, 0.06920816531554787, 0.09986991260234326]
    #config['gamma_i4'] = [0.08583027112988993, 0.05607691022676532, 0.05623253897381392, 0.04864864864864865, 0.03447715183551843]
    #config['delta_i3'] = [0.022156015184196173, 0.026321398431009075, 0.03880924059597372, 0.04047139614276389, 0.03191281403732929]
    #config['delta_i4'] = [0.0345187998184925, 0.042160957764474254, 0.07016475367157751, 0.06736083328317678, 0.10744122210287624]
    config["Exposed"] = [10, 10, 4, 3,3]
    config["Asymptomatic_Cases"] = [0, 0, 0, 0, 0]
    config["Mild_Cases"] = [0, 0, 0, 0, 0]
    config["Hospitalised_Cases"] = [0, 0, 0, 0, 0]
    config["ICU_Cases"] = [0, 0, 0, 0, 0]
    config["Recovered"] = [0, 0, 0, 0, 0]
    config["Deaths"] = [0, 0, 0, 0, 0]
    config["R0_lockdown_scale"] = 0.8

    # Wave 4

    # config['gamma_i3'] = [[0.001898, 64.5484], [0.001877, 67.53383], [0.002323, 51.79446],
    #                       [0.020177, 5.719088], [0.040358, 3.43061]]
    # config['gamma_i4'] = [[0.000795, 6.924638], [0.00082, 5.805063], [0.000443, 5.941322],
    #                       [0.001347, 3.705529], [0.00093, 1.211461]]
    # config['delta_i3'] = [[0.004772, 9.259029], [0.010461, 3.815892], [0.016261, 2.877455],
    #                       [0.003327, 15.41143], [0.022124, 1.236196]]
    # config['delta_i4'] = [[3.48E-06, 46353.52], [1.30E-05, 12689.14], [3.12E-06, 52661.49],
    #                       [3.97E-06, 40739.05], [0.027494, 5.987563]]

    #config['days'] = 250
    # immune_percentage = 0.58
    # config['Vaccinated'] = [immune_percentage * n for n in config['N']]
    # config['vacc_eff'] = 0.67
    # config['alpha_i24']= [0.0020833333333333333, 0.0011363636363636365, 0.0014285714285714286, 0.0009615384615384616, 0.0016129032258064516]
    # config['alpha_i23']= [0.002380952380952381, 0.0011904761904761906, 0.0029411764705882353, 0.000819672131147541, 0.0011111111111111111]
    #config['gamma_i3']= [0.1090734539246645, 0.07807242222689717, 0.08604199189123274, 0.07042976016229868, 0.1135561162388795]
    #config['gamma_i4']= [0.09433237047740836, 0.06694907717438482, 0.0668049611838433, 0.05293518832711232, 0.027821248478525476]
    #config['delta_i3']= [0.016203580951862015, 0.01663299833052414, 0.03692388689473671, 0.051940389626956796, 0.019956268221574345]
    #config['delta_i4']= [0.029152611499604898, 0.027928456238088255, 0.07036825007988072, 0.08066239316239317, 0.10571428571428572]
    # config["Exposed"] = [150, 150, 40, 50,50]
    # config["Asymptomatic_Cases"] = [0, 0, 0, 0, 0]
    # config["Mild_Cases"] = [0, 0, 0, 0, 0]
    # config["Hospitalised_Cases"] = [0, 0, 0, 0, 0]
    # config["ICU_Cases"] = [0, 0, 0, 0, 0]
    # config["Recovered"] = [0, 0, 0, 0, 0]
    # config["Deaths"] = [0, 0, 0, 0, 0]
    # config["R0_lockdown_scale"] = 0.9

    # Wave 5

    # config['gamma_i3'] = [[0.000514, 272.6185], [0.00257, 50.0909], [0.001409, 85.65876],
    #                       [0.01085, 10.55731], [0.001533, 92.72331]]
    # config['gamma_i4'] = [[0.008825, 0.201565], [0.002998, 0.182208], [0.001771, 0.132076],
    #                       [0, 0], [0, 0]]
    # config['delta_i3'] = [[0.019737, 1.352278], [0.053503, 0.720914], [0.014162, 4.337887],
    #                       [0, 0], [0.033131, 0.775107]]
    # config['delta_i4'] = [[1.10E-05, 14704.91], [1.33E-06, 124948.9], [1.00E-05, 15930.77],
    #                       [0, 0], [0, 0]]

    #config['days'] = 1000
    # immune_percentage = 0.9
    # config['Vaccinated'] = [immune_percentage * n for n in config['N']]
    # config['vacc_eff'] = 0.67
    # config['alpha_i24'] = [0.0009411764705882353, 0.0006857142857142859, 0.000631578947368421, 0.0004151515151515152, 0.002380952380952381]
    # config['alpha_i23'] = [0.0029411764705882353, 0.001724137931034483, 0.0038461538461538464, 0.0017857142857142859, 0.002]
    #config['gamma_i3'] = [0.09968969578526111, 0.06541712489882205, 0.07266953082875972, 0.06739544001853764, 0.09732992659801934]
    #config['gamma_i4'] = [0.0863104878271484, 0.05452800616825704, 0.05214207484072172, 0.04958343790745649, 0.029253975235468025]
    #config['delta_i3'] = [0.023374109239404476, 0.027547131827543325, 0.0374388708442654, 0.03807029983026096, 0.03162253780103785]
    #config['delta_i4'] = [0.0357202461844476, 0.043701628591555214, 0.06777358940713349, 0.06045035758005283, 0.11432644333570918]

    # config["Exposed"] = [150, 150, 40, 50,50]
    # config["Asymptomatic_Cases"] = [0, 0, 0, 0, 0]
    # config["Mild_Cases"] = [0, 0, 0, 0, 0]
    # config["Hospitalised_Cases"] = [0, 0, 0, 0, 0]
    # config["ICU_Cases"] = [0, 0, 0, 0, 0]
    # config["Recovered"] = [0, 0, 0, 0, 0]
    # config["Deaths"] = [0, 0, 0, 0, 0]
    # config["R0_lockdown_scale"] = 1
    # config["t_incubation"] = [1.5, 0.9]
    # config["t_infective"] = [1.5, 3]
    #config["mobility"]= [[0.8115834442, 0.0796074971, 0.0804403798, 0.013268029, 0.01510065],
    #                    [0.0289415763, 0.6365418334, 0.294823539, 0.0186296065, 0.0210634447],
    #                   [0.0300457299, 0.2856778345, 0.6191559164, 0.0326120901, 0.032508429],
    #                    [0.0090392005, 0.0284042544, 0.0485215909, 0.5427908922, 0.3712440619],
    #                    [0.0101938026, 0.0298346009, 0.0457389387, 0.3696105686, 0.5446220892]]



    #The with statement save the rates you put in a dictionary to json
    with open(os.path.join(MODEL_DIR, 'config_start.json'), 'w') as writer:
        json.dump(config, writer, indent=2)
        writer.close()
    #set-up the multi-SEIRDV model- This calculates all 5 SEIRDV models and lets them   interact with each other using the mobility matrix
    cmd = 'python3 main_multi.py'
    cmd += f" --config {os.path.join(MODEL_DIR, 'config_start.json')}"
    cmd += f" --config_out {os.path.join(MODEL_DIR, 'config_out_' + str(seed) + '.json')}"
    cmd += f" --data_out {os.path.join(MODEL_DIR, 'run_' + str(seed) + '.csv')}"

    os.system(cmd)

#execute
if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    data = []
    for seed in tqdm(SEEDS):
        run(seed)
        data.append(combine_subruns([f'{MODEL_DIR}/run_{seed}.csv']))
    
    t, xt, districts = combine_seeds(data)

    xt_var = xt.var(0)
    xt = xt.mean(0)

    comps = ['Susceptible','Vaccinated','Exposed','Asymptomatic','Mild','Hospitalised','Critical','Recovered','Deceased']
    districts = CONFIG['districts']
    cols = []
    for c in comps:
        for dist in districts:
            cols.append(f'{dist}_{c}')
    pop = np.array(CONFIG['N'])
    pop = pop.reshape(-1, 1, 1).repeat(len(comps), 1).repeat(xt.shape[-1], 2)
    
    xt = xt * pop
    xt_var = np.sqrt(xt_var) * pop
    
    xt = xt.T.reshape(xt.shape[-1], -1)
    xt = pd.DataFrame(xt, columns=cols)
    xt.to_csv(f'{MODEL_DIR}/means.csv', index=False)
    
    xt_var = xt_var.T.reshape(pop.shape[-1], -1)
    xt_var = pd.DataFrame(xt_var, columns=cols)
    xt_var.to_csv(f'{MODEL_DIR}/stds.csv', index=False)
