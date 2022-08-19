#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:28:15 2022

@author: s2113337

make new files of each bond info

"""

import pandas as pd
from tqdm import tqdm

filepath = '/home/s2113337/MAC-MIGS/IBM/'

#read in data
data = pd.read_csv(filepath+'bonds.csv').drop(columns='Unnamed: 0')


# function to extract rows for bond x 

def extract_bond(x, df):
    mask = df['label'] == x
    return df[mask]

#find unique bonds
bonds = data.label.unique()

# extract 

path = '/home/s2113337/MAC-MIGS/IBM/bond_files/'

for bond in tqdm(bonds):
    extracted_df = extract_bond(bond, data)
    extracted_df.to_csv(path+bond+'.csv')

