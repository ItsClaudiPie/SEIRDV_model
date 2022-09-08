import pandas as pd
from matplotlib import pyplot as plt

mean = [pd.read_excel(f'wave{i}_pred.xlsx', sheet_name='mean') for i in [1,2,3,4,5]]
lower = [pd.read_excel(f'wave{i}_pred.xlsx', sheet_name='lower') for i in [1,2,3,4,5]]
upper = [pd.read_excel(f'wave{i}_pred.xlsx', sheet_name='upper') for i in [1,2,3,4,5]]

mean = pd.concat(mean, 0).reset_index()
lower = pd.concat(lower, 0).reset_index()
upper = pd.concat(upper, 0).reset_index()

districts = ['City Of Tshwane Metro', 'City Of Johannesburg Metro', 'Ekurhuleni Metro', 'Sedibeng', 'West Rand']

colours = ['blue', 'orange', 'green', 'red', 'purple']
fig, axs = plt.subplots(len(districts))
width = 1.96
for i, dist in enumerate(districts):
    col = colours[i]

    key = f'New Cases {dist}'
    axs[i].plot(range(len(mean)), mean[key].rolling(window=7).mean(), label=f'{dist} Reported', color=col)

    key = f'{dist}_Mild'
    bottom = lower[key]
    top = upper[key]
    axs[i].fill_between(range(len(mean)), bottom.rolling(window=7).mean(),
                        top.rolling(window=7).mean(), color=col, alpha=0.3)
    axs[i].plot(range(len(mean)), mean[key].rolling(window=7).mean(), label=f'{dist} Predicted',
                color=col, linestyle='-.')

    axs[i].set_title(dist)
    axs[i].set_xticks([])

n_ticks = 45
plt.xticks(range(0, len(mean), n_ticks), mean['Date'][range(0, len(mean), n_ticks)])
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()

colours = ['blue', 'orange', 'green', 'red', 'purple']
fig, axs = plt.subplots(len(districts))
width = 1.96
for i, dist in enumerate(districts):
    col = colours[i]

    key = f'{dist} Hospitalised Cases'
    axs[i].plot(range(len(mean)), mean[key].rolling(window=7).mean(), label=f'{dist} Reported', color=col)

    key = f'{dist}_Hospitalised'
    bottom = lower[key]
    top = upper[key]
    axs[i].fill_between(range(len(mean)), bottom.rolling(window=7).mean(),
                        top.rolling(window=7).mean(), color=col, alpha=0.3)
    axs[i].plot(range(len(mean)), mean[key].rolling(window=7).mean(), label=f'{dist} Predicted',
                color=col, linestyle='-.')

    axs[i].set_title(dist)
    axs[i].set_xticks([])

plt.xticks(range(0, len(mean), n_ticks), mean['Date'][range(0, len(mean), n_ticks)])
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()
