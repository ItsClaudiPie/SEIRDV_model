import enum
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, uniform as uni_dist
import matplotlib.pyplot as plt

from model import SEIRDV, MultiDistrictModel
from run import RunModel


def load_config(path):
    with open(path, 'r') as configfile:
        config = json.load(configfile)
        configfile.close()
    
    return config


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--config_out', help='Config output file')
    parser.add_argument('--data_out', help='Data output file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set Seed for this run
    np.random.seed(config['seed'])

    # Define parameters for all districts in experiment
    t_incubation = [gamma_dist(a=config['t_incubation'][0], 
                scale=config['t_incubation'][1]).rvs()]*len(config['districts'])
    t_infective = [gamma_dist(a=config['t_infective'][0],
                scale=config['t_infective'][1]).rvs()]*len(config['districts'])
    
    config['Incubation_Period'] = t_incubation
    config['Infective_Period'] = t_infective
    
    vacc_eff = [config['vacc_eff']]*len(config['districts'])

    # Normalise and scale vulnerability index
    vulnerability_raw = [uni_dist(loc = config['vulnerability'][i][0],
                            scale=config['vulnerability'][i][1] - config['vulnerability'][i][0]).rvs()
                    for i in range(len(config['districts']))]
    vulnerability = np.array(vulnerability_raw)
    vulnerability = (vulnerability - vulnerability.mean()) / vulnerability.std() 
    vulnerability = vulnerability - vulnerability.min()
    vulnerability = config['vulnerability_range'] * vulnerability / vulnerability.max()
    vulnerability = vulnerability + config['vulnerability_start']

    # Generate national R0 and adjust for vulnerability
    R0_raw = np.array([gamma_dist(a=config['R0'][0],
                scale=config['R0'][1]).rvs()]*len(config['districts']))
    config['R0_lockdown_scale'] = uni_dist(loc = 0.6, scale = 0.4).rvs()
    R0 = (R0_raw * vulnerability * config['R0_lockdown_scale']).tolist()

    #Other transition rates
    gamma = 1/t_infective[0] #Assumption of days of infectiousness for asypmtomatic and mild cases
    rho = [config['rho']]*len(config['districts'])
    #p1 = [config['p1']]*len(config['districts'])

    p1 = [uni_dist(loc = 0.25, scale = 0.75).rvs()]*len(config['districts'])
    config['p1'] = p1
    
    gamma_i1 = [gamma]*len(config['districts']) #transition rate--  of recovering from asymptomatic
    
    alpha_i1  = [0.0 / uni_dist(loc = config['alpha_i1'][0],
                scale=config['alpha_i1'][1] - config['alpha_i1'][0]).rvs()]*len(config['districts']) #Assumed transition rate of asymptomatic case to mildy infected
    
    
    alpha_i2 = [uni_dist(loc = config['alpha_i2'][0],
                scale=config['alpha_i2'][1] - config['alpha_i2'][0]).rvs()]*len(config['districts']) #transition rate of mild case to General Ward (GW) -- data from hospitals parameters.word
    prop_i2 = [uni_dist(loc=max(r-0.05, 0), scale=0.1).rvs()
                for r in [0.119532785, 0.211992489, 0.180688433, 0.036078897, 0.041188556]]
    alpha_i2 = [prop_i2[i] / a for i, a in enumerate(alpha_i2)]
    #alpha_i2 = [0.002, 0.001724137931034483, 0.0017857142857142859, 0.0014705882352941176, 0.001282051282051282]

    gamma_i2 = [gamma]*len(config['districts']) #transition rate of mild case to recovered
    propg_i2 = [1-p for p in prop_i2]
    gamma_i2 = [propg_i2[i] * a for i, a in enumerate(gamma_i2)]
    gamma_i2 = [gamma]*len(config['districts'])
    
    alpha_i3  = [uni_dist(loc = config['alpha_i3'][0],
                scale=config['alpha_i3'][1] - config['alpha_i3'][0]).rvs()]*len(config['districts']) #transition rate  of GW to ICU-- data from hospitals parameters.word
    prop_i3 = [uni_dist(loc=max(r-0.05, 0), scale=0.1).rvs()
                for r in [0.162164729, 0.13106766, 0.107955885, 0.102029915, 0.047971245]]
    alpha_i3 = [prop_i3[i] / a for i, a in enumerate(alpha_i3)]
    #alpha_i3 = [0.002631578947368421, 0.0014285714285714286, 0.0029411764705882353, 0.0029411764705882353, 0.002272727272727273]
    
    delta_i3 = [uni_dist(loc = config['delta_i3'][0],
                scale=config['delta_i3'][1] - config['delta_i3'][0]).rvs()]*len(config['districts']) #transition rate of GW  to recovered -- data from hospitals parameters.word
    propd_i3 = [uni_dist(loc=max(r-0.05, 0), scale=0.1).rvs()
                for r in [0.112431133, 0.133728614, 0.166480677, 0.176878842, 0.174980033]]
    delta_i3 = [propd_i3[i] / a for i, a in enumerate(delta_i3)]
    #delta_i3 = [0.035294089749680076, 0.025617013984876785, 0.034901111158312556, 0.054351487928130264, 0.0429594891135747]

    gamma_i3 = [uni_dist(loc = config['gamma_i3'][0],
                scale=config['gamma_i3'][1] - config['gamma_i3'][0]).rvs()]*len(config['districts'])#transition rate of GW  to recovered -- data from hospitals parameters.word
    propg_i3 = [1 - p - propd_i3[i] for i, p in enumerate(prop_i3)]
    gamma_i3 = [propg_i3[i] / a for i, a in enumerate(gamma_i3)]
    #gamma_i3 = [0.05645849085422254, 0.05059842484263016, 0.05646333375560547, 0.06750861079219289, 0.07250426453807762]
   
     #transition rate of ICU cases to recovered
    delta_i4 = [uni_dist(loc = config['delta_i4'][0],
                scale=config['delta_i4'][1] - config['delta_i4'][0]).rvs()]*len(config['districts'])#transition rate of ICU cases to dead of covid - calculated rate using DATCOV19.csv
    propd_i4 = [uni_dist(loc=max(r-0.05, 0), scale=0.1).rvs()
                for r in [0.331732459, 0.336490646, 0.35501092, 0.493019197, 0.507204611]]
    delta_i4 = [propd_i4[i] / a for i, a in enumerate(delta_i4)]
    #delta_i4 = [0.0414777878513146, 0.04288552603679941, 0.03355315111797479, 0.05303689814504788, 0.08517215950989847] 

    gamma_i4 = [uni_dist(loc = config['gamma_i4'][0],
                scale=config['gamma_i4'][1] - config['gamma_i4'][0]).rvs()]*len(config['districts'])#transition rate of GW  to recovered -- data from hospitals parameters.word
    propg_i4 = [1 - p for p in propd_i4]
    gamma_i4 = [propg_i4[i] / a for i, a in enumerate(gamma_i4)] 
    #gamma_i4 = [0.05412956016264819, 0.03598792261818881, 0.056845630996921116, 0.06997124469396139, 0.045738030687482126]


    mu = [config['mu']]*len(config['districts']) # function, where we can switch on and off the transition rate for the feedback loop of recovered back to susceptible
    
    # Assuming 80% of exp remain in district and only 20% are mobile as mobility matrix doesnt include this method 3?? -- use method 4
    mobility = np.array(config['mobility']) # numbers from mobility.method3.rd

    config['mobility_level'] = [uni_dist(loc = 0, scale=1).rvs() for i in range(len(config['districts']))]
    mobility[range(len(mobility)), range(len(mobility))] = np.array(config['mobility_level'])

    mobility = mobility / mobility.sum(-1).reshape(-1,1)

    #vaccs rate
    upsilon = [uni_dist(loc = config['upsilon'][i][0],
                            scale=config['upsilon'][i][1] - config['upsilon'][i][0]).rvs()
                    for i in range(len(config['districts']))]
    
    config['Exposed'] = [uni_dist(loc = 1, scale=19).rvs() for i in range(len(config['districts']))]
    immune_percentage = uni_dist(loc = 0, scale=60).rvs() / 100
    config['Vaccinated'] = [immune_percentage * n for n in config['N']]

    config['Exposed_start'] = deepcopy(config['Exposed'])
    config['Vaccinated_start'] = deepcopy(config['Vaccinated'])
    config['Immunity_start'] = immune_percentage
    
    # Define a SEIRDV model for each of the districts defined
    seir = [SEIRDV(R0[i], t_infective[i], upsilon[i], vacc_eff[i], rho[i],
                p1[i], t_incubation[i], gamma_i1[i], gamma_i2[i], gamma_i3[i],
                gamma_i4[i], alpha_i1[i], alpha_i2[i], alpha_i3[i], delta_i3[i],
                delta_i4[i], config['N'][i], mu[i]) for i in range(len(config['N']))]

    # Create a MultiDistrict run setup using the mobility matrix
    seir = MultiDistrictModel(seir, mobility, 2)
    
    # Run model for 500 days
    seir = RunModel(seir)
    t, xt = seir.run(days=config['days'], vac_0=config['Vaccinated'],
            exp_0=config['Exposed'], asymp_0=config['Asymptomatic_Cases'],
            mild_0=config['Mild_Cases'], hosp_0=config['Hospitalised_Cases'],
            icu_0=config['ICU_Cases'], rec_0=config['Recovered'],
            die_0=config['Deaths'])
    
    # Reshape output data to format [n_districts, n_compartments, n_days]
    xt = np.concatenate([x.reshape(-1, 1) for x in xt], 1).reshape(len(t), len(config['N']), -1)
    xt = xt.transpose((1, 2, 0))

    data = np.concatenate((t.reshape(-1, 1), xt.reshape(-1, xt.shape[-1]).T), -1)
    end_compartments = data[-1][1:].reshape(-1, 9).T
    end_compartments[end_compartments < 0] = 0.0
    end_compartments = end_compartments * np.array(config['N'])

    columns = [[f'{d}_Susceptable', f'{d}_Vaccinated', f'{d}_Exposed',
                f'{d}_Asymptomatic', f'{d}_Mild', f'{d}_Hospitalised',
                f'{d}_ICU', f'{d}_Recovered', f'{d}_Deaths']
                for d in config['districts']]
    columns = ['Time'] + [c for l in columns for c in l]

    data = pd.DataFrame(data, columns=columns)
    data.to_csv(args.data_out, index=False)
    
    config['Vaccinated'] = end_compartments[1].tolist()
    config['Exposed'] = end_compartments[2].tolist()
    config['Asymptomatic_Cases'] = end_compartments[3].tolist()
    config['Mild_Cases'] = end_compartments[4].tolist()
    config['Hospitalised_Cases'] = end_compartments[5].tolist()
    config['ICU_Cases'] = end_compartments[6].tolist()
    config['Recovered'] = end_compartments[7].tolist()
    config['Deaths'] = end_compartments[8].tolist()

    config['vulnerability'] = vulnerability_raw
    config['vulnerability_normalised'] = vulnerability.tolist()
    config['upsilon'] = upsilon

    config['gamma_i1'] = gamma_i1 
    config['gamma_i2'] = gamma_i2 
    config['gamma_i3'] = gamma_i3
    config['gamma_i4'] = gamma_i4
    config['alpha_i1'] = alpha_i1
    config['alpha_i2'] = alpha_i2
    config['alpha_i3'] = alpha_i3
    config['delta_i3'] = delta_i3
    config['delta_i4'] = delta_i4

    config['R0'] = R0_raw.tolist()
    config['Adjusted_R0'] = R0

    with open(args.config_out, 'w') as configfile:
        json.dump(config, configfile, indent=2)
