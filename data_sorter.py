
#split dataset into different geometries (ts, r, rcc, rcu, pc)

import pandas as pd
import numpy as np
from tqdm import tqdm

path = '/home/s1997751/Documents/PhD/Year2/ibm_project/data_files/qmrxn/'
energies = pd.read_csv(path+'energies.txt')

energies['filename'] = energies['filename'].apply([lambda x: str(x)[:-3]+'soap'])

rcc_path = "reactant-complex-constrained-conformers/sn2/"
rcu_path = "reactant-complex-unconstrained-conformers/sn2/"
ts_path = "transition-states/sn2/"
pc_path = "product-conformers/sn2/"

ts = energies.loc[energies['geometry'] == 'ts'].reset_index(drop=True)
ts = ts.loc[ts['reaction'] == 'sn2'].reset_index(drop=True)
ts = ts.loc[ts['method'] == 'mp2'].reset_index(drop=True)

# r = energies.loc[energies['geometry'] == 'r'].reset_index(drop=True)
# r = r.loc[r['reaction'] == 'sn2'].reset_index(drop=True)

rcc = energies.loc[energies['geometry'] == 'rcc'].reset_index(drop=True)
rcc = rcc.loc[rcc['reaction'] == 'sn2'].reset_index(drop=True)
rcc = rcc.loc[rcc['method'] == 'mp2'].reset_index(drop=True)

rcu = energies.loc[energies['geometry'] == 'rcu'].reset_index(drop=True)
rcu = rcu.loc[rcu['reaction'] == 'sn2'].reset_index(drop=True)
rcu = rcu.loc[rcu['method'] == 'mp2'].reset_index(drop=True)

pc = energies.loc[energies['geometry'] == 'pc'].reset_index(drop=True)
pc = pc.loc[pc['reaction'] == 'sn2'].reset_index(drop=True)
pc = pc.loc[pc['method'] == 'mp2'].reset_index(drop=True)
                                                                           
# print(ts)

# rcc_splits = np.array([i.rsplit("_",2) for i in rcc['label'].values])
# rcu_splits = np.array([i.rsplit("_",2) for i in rcu['label'].values])
pc_splits = np.array([i.rsplit("_",2) for i in pc['label'].values])

# print(rcc_splits.shape)
# print(rcu_splits.shape)
# print(pc_splits.shape)
# print(pc_splits[:,[0,2]])
# print(rcu['label'].values[:10])


# tuple_list = np.array(['a','a','a'])
tuple_list = np.array(['a','a'])

# curr_tuple = np.zeros((3,1))

for index, mol in tqdm(enumerate(ts['label']), total=ts.shape[0]):
    # if index==77:
    elem = np.array(mol.rsplit("_",2))
    # print(elem)
    # print(elem[[0,2]])
    
    
    # check_pc = np.isin(pc_splits[:,[0,2]], elem[[0,2]])
    # pos_pc = np.all(check_pc,axis=1)
    # print(np.sum(pos_pc))
    
    pos_rcc = np.isin(rcc['label'].values, mol)
    # print(np.sum(pos_rcc))
    
    pos_rcu = np.isin(rcu['label'].values, mol)
    # print(pos_rcu)
    # print(ts.iloc[index].filename)

    # curr_tuple[2,0] = mol
    # curr_tuple = np.hstack((np.where(pos_rcc, rcc['label'].values)))
    # print(rcc['filename'].values[np.where(pos_rcc)])
    # print(rcu['filename'].values[np.where(pos_rcu)])
    # print(pc['filename'].values[np.where(pos_pc)])
    
    
    
    # for i in range(np.sum(pos_rcc)):
    #     for j in range(np.sum(pos_pc)):
    #         curr_tuple = np.array([rcc['filename'].values[np.where(pos_rcc)][i], pc['filename'].values[np.where(pos_pc)][j], ts.iloc[index].energy])
    #         tuple_list = np.vstack((tuple_list, curr_tuple))
            
    # for i in range(np.sum(pos_rcu)):
    #     for j in range(np.sum(pos_pc)):
    #         curr_tuple = np.array([rcu['filename'].values[np.where(pos_rcu)][i], pc['filename'].values[np.where(pos_pc)][j], ts.iloc[index].energy])
    #         tuple_list = np.vstack((tuple_list, curr_tuple))     


    for i in range(np.sum(pos_rcc)):
        curr_tuple = np.array([rcc['filename'].values[np.where(pos_rcc)][i], ts.iloc[index].energy])
        tuple_list = np.vstack((tuple_list, curr_tuple))
            
    for i in range(np.sum(pos_rcu)):
        curr_tuple = np.array([rcu['filename'].values[np.where(pos_rcu)][i], ts.iloc[index].energy])
        passtuple_list = np.vstack((tuple_list, curr_tuple))      
            
print(tuple_list.shape)
tuple_list= np.delete(tuple_list,0,0)

np.save("tuple_list_noprod.npy", tuple_list)
    