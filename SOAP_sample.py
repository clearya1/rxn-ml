from tqdm import tqdm
import numpy as np
import pandas as pd
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix

from ase import Atoms



# Define atomic structures


#read our data
sample_df = pd.read_csv('/home/s2122199/Documents/Edinburgh/projects/IBM/rxn-ml/dataset/energies_coordinates.csv')
#sample_df = sample_df.head()



#parameters for soap, see https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
species=["C", "H", "F",  "Cl", "N", "Br", "O"]
rcut = 5 
nmax = 8
lmax = 6

# creates soap descriptor (local), will be an 2D array, 1 dimension = #atoms in molecule
soap_desc = SOAP(species=species,rcut=5, nmax=8, lmax=6) #, crossover=True)

# creates a global descriptor (average), 1 dimensional, same shape for all molecules.
average_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="inner",
    sparse=False
)

#create soap descriptor

#use the first 2 atoms of the data set
atoms1 = Atoms('H6C2F', [(sample_df['x coordinates_0'][0], sample_df['y coordinates_0'][0], sample_df['z coordinates_0'][0]), (sample_df['x coordinates_1'][0], sample_df['y coordinates_1'][0], sample_df['z coordinates_1'][0]),(sample_df['x coordinates_2'][0], sample_df['y coordinates_2'][0], sample_df['z coordinates_2'][0]),(sample_df['x coordinates_3'][0], sample_df['y coordinates_3'][0], sample_df['z coordinates_3'][0]),(sample_df['x coordinates_4'][0], sample_df['y coordinates_4'][0], sample_df['z coordinates_4'][0]),(sample_df['x coordinates_5'][0], sample_df['y coordinates_5'][0], sample_df['z coordinates_5'][0]),(sample_df['x coordinates_6'][0], sample_df['y coordinates_6'][0], sample_df['z coordinates_6'][0]),(sample_df['x coordinates_7'][0], sample_df['y coordinates_7'][0], sample_df['z coordinates_7'][0]),(sample_df['x coordinates_8'][0], sample_df['y coordinates_8'][0], sample_df['z coordinates_8'][0])])

atoms2 = Atoms('FH6C', [(sample_df['x coordinates_0'][1], sample_df['y coordinates_0'][1], sample_df['z coordinates_0'][1]), (sample_df['x coordinates_1'][1], sample_df['y coordinates_1'][1], sample_df['z coordinates_1'][1]),(sample_df['x coordinates_2'][1], sample_df['y coordinates_2'][1], sample_df['z coordinates_2'][1]),(sample_df['x coordinates_3'][1], sample_df['y coordinates_3'][1], sample_df['z coordinates_3'][1]),(sample_df['x coordinates_4'][1], sample_df['y coordinates_4'][1], sample_df['z coordinates_4'][1]),(sample_df['x coordinates_5'][1], sample_df['y coordinates_5'][1], sample_df['z coordinates_5'][1]),(sample_df['x coordinates_6'][1], sample_df['y coordinates_6'][1], sample_df['z coordinates_6'][1]),(sample_df['x coordinates_7'][1], sample_df['y coordinates_7'][1], sample_df['z coordinates_7'][1])])


soap1 = soap_desc.create(atoms1) #, positions=[0])
soap2 = soap_desc.create(atoms2) 

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
        
   
    
np.save('/home/s2122199/Documents/Edinburgh/projects/IBM/rxn-ml/dataset/soap_transition.npy',descriptor)

