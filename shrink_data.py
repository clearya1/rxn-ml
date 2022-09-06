#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 17:38:49 2022

@author: aidan
"""

import numpy as np
from tqdm import tqdm
from os import listdir
from pathlib import Path


PATH = "/home/aidan/Documents/PhD/Year2/ibm_project/data_files/qmrxn/bonds_files/"
folders = [PATH+temp for temp in listdir(PATH)]
names = listdir(PATH)


for i in tqdm(range(len(names)), total=len(names)):
    txtfile = np.loadtxt(folders[i], dtype=object, delimiter=',')

    bond_dir = PATH+names[i][:-4]+"/"
    if not Path(bond_dir).exists():
        Path(bond_dir).mkdir(parents=True)
    for j in range(txtfile.shape[0]):
        np.save(bond_dir+"sample_"+str(j)+".npy", txtfile[j, :])

    info_file = np.array([str(bond_dir+"sample_"+str(k)+".npy") for k in range(txtfile.shape[0])])
    np.save(bond_dir+names[i][:-4]+".npy", info_file)
