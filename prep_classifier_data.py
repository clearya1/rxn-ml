from tqdm import tqdm
import numpy as np
import pandas as pd
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix

from ase import Atoms



# Define atomic structures


#read our data
sample_df = pd.read_csv('/Users/Andrew/Documents/Edinburgh/Chemisty ML/energies_coordinates.csv')



#parameters for soap, see https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
species=["C", "H", "F",  "Cl", "N", "Br", "O"]
rcut = 5 
nmax = 8
lmax = 6


# creates a global descriptor (average), 1 dimensional, same shape for all molecules.
average_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="inner",
    sparse=False
)


# empty dataframe to save SOAP descritors and binary if transition state or not
descriptor = np.empty((len(sample_df.index),2),dtype=object)

element_columns = sample_df.columns[7::4]
x_columns = sample_df.columns[8::4]
y_columns = sample_df.columns[9::4]
z_columns = sample_df.columns[10::4]


for i, row in tqdm(sample_df.iterrows(),total = len(sample_df.index)):
        
    #create chemical name of molecule
    #create mask for where we have atoms
        
    mask = row[element_columns].notna() #  == nan 
    mol_name = ''.join(row[element_columns][mask])
        
    #create array containing the positions of the atoms
    pos = np.empty((3,np.sum(mask)))
    pos[0] = row[x_columns][:np.sum(mask)]
    pos[1] = row[y_columns][:np.sum(mask)]
    pos[2] = row[z_columns][:np.sum(mask)]
        
    pos = np.transpose(pos)
    
    atom = Atoms(mol_name,pos)
        
    descriptor[i][0] = average_soap.create(atom)
        
    #if transition state, then = 1
    if sample_df['geometry'][i] == 'ts':
        descriptor[i][1] = 1
    else:
        descriptor[i][1] = 0
        
   
    
np.save('/Users/Andrew/Documents/Edinburgh/Chemisty ML/soap_transition.npy',descriptor)

