import os
import numpy as np
import pickle 
from tqdm import tqdm

import matplotlib.pyplot as plt


def moving_average(x, N=100):
    return np.convolve(x, np.ones(N) / N, mode='valid')

ENV = 'driving'

_, ax = plt.subplots(1, 1, figsize=(5, 5))

pomdp_runs_loc = f'./{ENV}/data'

pomdp_runs = [f for f in os.listdir(pomdp_runs_loc)]

pomdp_data = []
for p in pomdp_runs:
    loc = os.path.join(pomdp_runs_loc, p)
    data = np.genfromtxt(loc, delimiter=',')
    pomdp_data.append(data)
pomdp_data = np.stack(pomdp_data)

sr_pomdp = pomdp_data[:, :, 1:3]
sh_pomdp = pomdp_data[:, :, 3:5]

count_collisions_pomdp = []
car_width = 1
for i in range(len(sr_pomdp)):
    start_idxs = np.where((sr_pomdp[i, :, 1] == 0.))[0]

    sr_pomdp_run = sr_pomdp[i, start_idxs + 9, 0]
    sh_pomdp_run = sh_pomdp[i, start_idxs + 9, 0]

    count_collisions_pomdp_run = np.where(abs(sh_pomdp_run - sr_pomdp_run) < car_width, 1, 0)
    count_collisions_pomdp_run = count_collisions_pomdp_run.sum()

    count_collisions_pomdp.append(count_collisions_pomdp_run)

results = {'pomdp': count_collisions_pomdp}
with open('./results_collisions_driving.pkl', 'wb') as f:
    pickle.dump(results, f)

ax.bar([-1, 1], [np.mean(count_collisions_pomdp)])
ax.errorbar([-1, 1], [np.mean(count_collisions_pomdp)], [np.std(count_collisions_pomdp)], capsize=3, linestyle='')
ax.set_xticks([-1, 1])
ax.set_xticklabels(['POMDP'])
ax.set_ylabel('# of collisions')

plt.savefig('./Figures/count_collisions_driving.png', dpi=600)
plt.savefig('./Figures/count_collisions_driving.svg', dpi=600)
