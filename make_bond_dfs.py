#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:28:15 2022

@author: s2113337

make new files of each bond info

"""

import pandas as pd
import numpy as np
from tqdm import tqdm

filepath = '/home/s2113337/MAC-MIGS/IBM/'
filepath = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/'


#read in data
data = np.load(filepath+'bonds.npy', allow_pickle=True)
#data = pd.read_csv(filepath+'bonds.csv').drop(columns='Unnamed: 0')

data = pd.DataFrame(data)
data = data.rename(columns = {0: 'label', 561: 'dist'})     

# function to extract rows for bond x 

def extract_bond(x, df):
    mask = df['label'] == x
    return df[mask]

#find unique bonds
bonds = data.label.unique()

# extract 

path = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/bond_files/'

for bond in tqdm(bonds):
    extracted_df = extract_bond(bond, data)
    extracted_df.to_csv(path+bond+'.csv')

