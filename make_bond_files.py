#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 21:10:47 2022

@author: s2122199
"""
import numpy as np
from dscribe.descriptors import ACSF
from ase import Atoms
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#path to .soap/ xyz files + where the master file will be saved
xyz_dir = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/'
tuple_file = '/home/s2122199/Documents/Edinburgh/projects/IBM/rxn-ml/file_lists/tuple_list.npy'

bond_dir = xyz_dir + 'bonds_files_withprod/'

if not Path(bond_dir).exists():
            Path(bond_dir).mkdir(parents = True)
# path to file with tuples


species=["C", "H", "F",  "Cl", "N", "Br", "O"]
        
acsf = ACSF(
    species=species,
    rcut=11.0,
    g2_params = [[16, 0.9], [16, 1.5], [16, 2.0], [16, 2.5], [16, 3.0], [16, 3.5], [16, 4.0], [16, 4.5], [16, 5.0], [16, 5.5], [16, 6.0], [16, 6.5], [16, 7.0], [16, 7.5], [16, 8.0], [16, 8.5], [16, 9.0], [16, 9.5], [16, 10.0], [16, 10.5], [16, 11.0], [16, 11.5], [16, 12.0]],

    g4_params = [[8.0, 16.0, 1.0], [8.0, 16.0, 2.0], [8.0, 32.0, 1.0], [8.0, 32.0, 2.0]]
    )
    


     
files = np.load(tuple_file)


for idx in tqdm(range(len(files)),total = len(files)):
        
        coords = pd.read_csv(str(xyz_dir) + str(files[idx][0])[:-4] + 'xyz', skiprows = 2, delim_whitespace = True, header = None)
        mol_name = ''.join(coords[0])
        pos = np.array(coords[[1,2,3]])
        atom = Atoms(mol_name,pos)
        reactant = acsf.create(atom, positions = np.arange(len(coords[0])))
        
        
        ##not including product at the moment        
        coords = pd.read_csv(str(xyz_dir) + str(files[idx][1])[:-4] + 'xyz', skiprows = 2, delim_whitespace = True, header = None)
        mol_name = ''.join(coords[0])
        pos = np.array(coords[[1,2,3]])
        atom = Atoms(mol_name,pos)
        product = acsf.create(atom, positions = np.arange(len(coords[0])))
                    
        coords = pd.read_csv(str(xyz_dir) + str(files[idx][1])[:-4] + 'xyz', skiprows = 2, delim_whitespace = True, header = None)
        
        #for use with product
        #coords = pd.read_csv(str(xyz_dir) + str(files[idx][2])[:-4] + 'xyz', skiprows = 2, delim_whitespace = True, header = None)
        mol_name = ''.join(coords[0])
        pos = np.array(coords[[1,2,3]])
        atoms  = coords[0]
        
        
        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms[i:]):
                dist = np.linalg.norm(pos[i]-pos[j])
                label = ''.join(np.sort([atom1, atom2]))
            
                vector_concat = np.append(reactant[i], reactant[i+j] ) #could include product here as well (change dim of array def at the beginning)
                vector_prod = np.append(product[i], product[i+j])
                vector_concat = np.append(vector_concat, vector_prod)
                arr = np.append(label, np.append(vector_concat, dist))
                with open(bond_dir+ str(label)+ '.txt','a') as bondfile:
                    np.savetxt(bondfile, [arr], fmt='%s',delimiter = ',')
                    bondfile.close()


