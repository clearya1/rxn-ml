# Code to compute the every interatomic distance for each pair of atom types in the data set.
# Also contains code to create histograms for the distribution of these interatomic distances.

# split dataset into different geometries (ts, r, rcc, rcu, pc)

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# %%
path = '/home/s1997751/Documents/PhD/Year2/ibm_project/data_files/qmrxn/'
energies = pd.read_csv(path+'energies_coordinates.csv', low_memory=(False))
energies = energies.loc[energies['reaction'] == 'sn2'].reset_index(drop=True)
energies = energies.loc[energies['method'] == 'mp2'].reset_index(drop=True)
energies.drop(energies.columns[:7], axis=1, inplace=True)

# %%


# global distance_matrix
distance_matrix = np.zeros((1, 2))


def pairwise_distance(row):
    global distance_matrix

    atoms = row.filter(regex='element')
    coords = row.filter(regex='coordinates')

    atoms = atoms.to_numpy(dtype=str)
    coords = coords.to_numpy(dtype=float)

    atoms = atoms[np.where(atoms != 'nan')]
    coords = coords[~np.isnan(coords)]

    coords = coords.reshape(-1, 3)

    for i, atom1 in enumerate(atoms):
        for j, atom2 in enumerate(atoms[i+1:], start=i+1):
            dist = np.linalg.norm(coords[i, :]-coords[j, :])
            label = ''.join(np.sort([atom1, atom2]))
            vec_out = np.array([label, dist])
            distance_matrix = np.vstack((distance_matrix, vec_out))


# # %%

# chunks = np.arange(0, energies.shape[0], 50)


# for idx in tqdm(range(1, len(chunks)),
#                 total=len(chunks),
#                 desc='Computing distances'):
#     energies.iloc[chunks[idx-1]:chunks[idx]].apply(pairwise_distance, axis=1)
#     np.save('/home/s1997751/Documents/PhD/Year2/ibm_project/dist_mats'
#             '/dist_mat'+str(idx)+'.npy', distance_matrix)
#     distance_matrix = np.zeros((1, 2))


# # %%
# chunks = np.arange(0, energies.shape[0], 50)
# distance_matrix = np.zeros((1, 2))

# for idx in tqdm(range(1, len(chunks)), total=len(chunks), desc='Loading distances'):
#     temp = np.load('/home/s1997751/Documents/PhD/Year2/ibm_project/dist_mats/dist_mat'+str(idx)+'.npy')[1:]
#     distance_matrix = np.concatenate((distance_matrix, temp))
# distance_matrix = distance_matrix[1:]
# np.save('/home/s1997751/Documents/PhD/Year2/ibm_project/dist_mats/dist_mat_master.npy', distance_matrix)

# %%

dist_mat = np.load('/home/aidan/Documents/PhD/Year2/ibm_project/dist_mats/dist_mat_master.npy')

data = pd.DataFrame(dist_mat, columns=['label', 'dist'])
data['dist'] = data['dist'].apply(pd.to_numeric)

# %%
# print(data.tail())
plt.rcParams['font.size'] = '26'

labels = data.label.unique()
grouped_data = data.groupby(by='label')
fig, ax = plt.subplots(figsize=(16,12))
for lab in labels:
    print(lab)
    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(grouped_data.get_group(lab).dist, bins=25,ax=ax).set(title=lab)
    # plt.show()
    plt.savefig(lab+".png")
    # grouped_data.get_group(lab).plot.hist(column=['dist'])
fig, ax = plt.subplots(figsize=(16,12))

sns.histplot(data.dist, bins=25,ax=ax).set(title="All distances")
plt.savefig("all_dist.png")

