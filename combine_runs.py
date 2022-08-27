import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def combine_subruns(paths):
    data = [pd.read_csv(p) for p in paths]
    data = [d[1:] if i > 0 else d for i, d in enumerate(data)]

    data = pd.concat(data)
    data['Time'] = np.linspace(0, len(data)-1, len(data))

    return data

def combine_seeds(data):
    t = data[0]['Time'].values
    
    xt = [d.values[:, 1:].T for d in data]

    dists = [k for k in data[0].keys()][1:]
    dists = [v.split('_', 1)[0] for v in dists]
    dists = list(set(dists))

    xt = [x.reshape(1, len(dists), -1, x.shape[-1]) for x in xt]
    xt = np.concatenate(xt, 0)

    return t, xt, dists


if __name__ == "__main__":
    dir = 'SENS_21_res'
    # paths = [f'{dir}/run1{i}.csv' for i in [1,2]]
    #paths = [f'{dir}/run_1.csv']
    #data1 = combine_subruns(paths)

    # paths = [f'{dir}/run2{i}.csv' for i in [1,2]]
    #paths = [f'{dir}/run_2.csv']
    #data2 = combine_subruns(paths)

    # paths = [f'{dir}/run3{i}.csv' for i in [1,2]]
    #paths = [f'{dir}/run_3.csv']
    #data3 = combine_subruns(paths)

    # paths = [f'{dir}/run4{i}.csv' for i in [1,2]]
    #paths = [f'{dir}/run_4.csv']
    #data4 = combine_subruns(paths)

    # paths = [f'{dir}/run5{i}.csv' for i in [1,2]]
    #paths = [f'{dir}/run_5.csv']
    #data5 = combine_subruns(paths)

    data = [combine_subruns([f'{dir}/run_{i}.csv']) for i in [1,2,3,4,5]]
    t, xt, districts = combine_seeds(data)

    #t, xt, districts = combine_seeds([data1, data2, data3, data4, data5])

    top = np.percentile(xt, (10, 90), 0)
    bottom = top[0]
    top = top[1]
    xt_var = xt.var(0)
    xt = xt.mean(0)

    comps = ['Susceptible','Vaccinated','Exposed','Asymptomatic','Mild','Hospitalised','Critical','Recovered','Deceased']
    districts = ["City Of Tshwane Metro", "City Of Johannesburg Metro", "Ekurhuleni Metro", "Sedibeng", "West Rand"]
    cols = []
    for c in comps:
        for dist in districts:
            cols.append(f'{dist}_{c}')
    # pop = np.array([3522325, 5538596, 3781377, 952102, 922640])
    # pop = pop.reshape(-1, 1, 1).repeat(9, 1).repeat(131, 2)
    #
    # xt = xt * pop
    # xt_var = np.sqrt(xt_var) * pop
    # # top = xt + 2.7 * xt_var
    # # bottom = xt - 2.7 * xt_var
    # # bottom = np.max(bottom, 0)
    # top *= pop
    # bottom *= pop
    #
    # xt = xt.T.reshape(131,-1)
    # # xt = xt.reshape(93, len(districts), len(comps)).transpose(1,2,0)
    # xt = pd.DataFrame(xt, columns=cols)
    # xt.to_csv(f'{dir}/means.csv', index=False)
    #
    # xt_var = xt_var.T.reshape(131,-1)
    # # xt_var = np.sqrt(xt_var)
    # # xt = xt.reshape(93, len(districts), len(comps)).transpose(1,2,0)
    # xt_var = pd.DataFrame(xt_var, columns=cols)
    # xt_var.to_csv(f'{dir}/stds.csv', index=False)
    # quit()

    # peaks = xt[:, [3,4,5,6], :].sum(1).argmax(1)
    # pop = [3522325, 5538596, 3781377, 952102, 922640]
    # districts = ["City Of Tshwane Metro", "City Of Johannesburg Metro", "Ekurhuleni Metro", "Sedibeng", "West Rand"]
    # for i in range(5):
    #     print(districts[i])
    #     for j, subset in enumerate(['Susceptible','Vaccinated','Exposed','Asymptomatic','Mild','Hospitalised','Critical','Recovered','Deceased']):
    #         print(subset, xt[i, j, peaks[i]]*pop[i])
    #     print('Cases', xt[i, [3,4,5,6], peaks[i]].sum(-1)*pop[i])

    #for i, dist in enumerate(districts):
    #    plt.plot(t, xt[i, 4], lw=3, label=dist)
    #    #plt.fill_between(t, bottom[i, 4], top[i, 4], alpha=0.3)
    #plt.legend()
    #plt.show()
    #quit()

    # Plot the output data
    plt.figure(figsize=(12, 7))

    plt.subplot(3, 3, 1)
    plt.title('Susceptible')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 0], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 0], top[i, 0], alpha=0.3)
    plt.ylabel('Fraction')

    plt.subplot(3, 3, 2)
    plt.title('Vaccinated')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 1], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 1], top[i, 1], alpha=0.3)

    plt.subplot(3, 3, 3)
    plt.title('Exposed')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 2], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 2], top[i, 2], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.15, 1.1))

    plt.subplot(3, 3, 4)
    plt.title('Asymptomatic Cases')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 3], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 3], top[i, 3], alpha=0.3)
    plt.ylabel('Fraction')

    plt.subplot(3, 3, 5)
    plt.title('Mild Cases')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 4], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 4], top[i, 4], alpha=0.3)

    plt.subplot(3, 3, 6)
    plt.title('Hospitalised Cases')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 5], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 5], top[i, 5], alpha=0.3)

    plt.subplot(3, 3, 7)
    plt.title('Critical(ICU) Cases')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 6], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 6], top[i, 6], alpha=0.3)
    plt.ylabel('Fraction')
    plt.xlabel('Time')

    plt.subplot(3, 3, 8)
    plt.title('Recovered')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 7], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 7], top[i, 7], alpha=0.3)
    plt.xlabel('Time')

    plt.subplot(3, 3, 9)
    plt.title('Deceased')
    for i, dist in enumerate(districts):
        plt.plot(t, xt[i, 8], lw=3, label=dist)
        plt.fill_between(t, bottom[i, 8], top[i, 8], alpha=0.3)
    plt.xlabel('Time')

    plt.show()
