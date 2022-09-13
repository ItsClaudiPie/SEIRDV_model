import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

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
    
    vacc_eff = [config['vacc_eff']]*len(config['districts'])

    vacc_rate = [gamma_dist(a=config['upsilon'][i][0], scale=config['upsilon'][i][1]).rvs() if config['upsilon'][i][0] else 0.0
                 for i in range(len(config['districts']))]

    # Normalise and scale vulnerability index
    vulnerability = np.array(config['vulnerability'])
    vulnerability = (vulnerability - vulnerability.mean()) / vulnerability.std() 
    vulnerability = vulnerability - vulnerability.min()
    vulnerability = config['vulnerability_range'] * vulnerability / vulnerability.max()
    vulnerability = vulnerability + config['vulnerability_start']

    # Generate national R0 and adjust for vulnerability
    R0 = np.array([gamma_dist(a=config['R0'][0],
                scale=config['R0'][1]).rvs()]*len(config['districts']))
    R0 = (R0 * vulnerability * config['R0_lockdown_scale']).tolist()

    #Other transition rates
    gamma = 1/t_infective[0] #Assumption of days of infectiousness for asypmtomatic and mild cases
    rho = [config['rho']]*len(config['districts'])
    p1 = [config['p1']]*len(config['districts'])
    gamma_i1 = [gamma]*len(config['districts']) #transition rate--  of recovering from asymptomatic
    gamma_i2 = [gamma]*len(config['districts']) #transition rate of mild case to recovered
    alpha_i23 = config['alpha_i23'] #transition rate of mild case to General Ward (GW) -- data from hospitals parameters.word
    gamma_i3 = [gamma_dist(a=config['gamma_i3'][i][0], scale=config['gamma_i3'][i][1]).rvs() if config['gamma_i3'][i][0] else 0.0
                for i in range(len(config['districts']))]
    alpha_i24 = config['alpha_i24'] #transition rate  of GW to ICU-- data from hospitals parameters.word
    delta_i3 = [gamma_dist(a=config['delta_i3'][i][0], scale=config['delta_i3'][i][1]).rvs() if config['delta_i3'][i][0] else 0.0
                for i in range(len(config['districts']))]
    gamma_i4 = [gamma_dist(a=config['gamma_i4'][i][0], scale=config['gamma_i4'][i][1]).rvs() if config['gamma_i4'][i][0] else 0.0
                for i in range(len(config['districts']))]
    delta_i4 = [gamma_dist(a=config['delta_i4'][i][0], scale=config['delta_i4'][i][1]).rvs() if config['delta_i4'][i][0] else 0.0
                for i in range(len(config['districts']))]
    mu = [config['mu']]*len(config['districts']) # function, where we can switch on and off the transition rate for the feedback loop of recovered back to susceptible
    
    # Assuming 80% of exp remain in district and only 20% are mobile as mobility matrix doesnt include this method 3?? -- use method 4
    mobility = np.array(config['mobility']) # numbers from mobility.method3.rd
    
    # Define a SEIRDV model for each of the districts defined
    seir = [SEIRDV(R0[i], t_infective[i], vacc_rate[i], vacc_eff[i], rho[i],
                   p1[i], t_incubation[i], gamma_i1[i], gamma_i2[i], gamma_i3[i],
                   gamma_i4[i], alpha_i23[i], alpha_i24[i], delta_i3[i],
                   delta_i4[i], config['N'][i]) for i in range(len(config['N']))]

    # Create a MultiDistrict run setup using the mobility matrix
    seir = MultiDistrictModel(seir, mobility, 2)
    
    # Run model for 500 days
    seir = RunModel(seir)
    t, xt = seir.run(days=config['days'], vac_0=config['Vaccinated'],
                     exp_0=config['Exposed'], asymp_0=config['Asymptomatic_Cases'],
                     mild_0=config['Mild_Cases'], hosp_0=config['Hospitalised_Cases'],
                     icu_0=config['ICU_Cases'], rec_0=config['Recovered'], die_0=config['Deaths'])
    
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

    with open(args.config_out, 'w') as configfile:
        json.dump(config, configfile, indent=2)
