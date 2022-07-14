
import numpy as np
import pandas as pd
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix

from ase import Atoms



# Define atomic structures


#read our data
sample_df = pd.read_csv('/home/s2122199/Documents/Edinburgh/projects/IBM/rxn-ml/dataset/energies_coordinates.csv')
sample_df = sample_df.head()

#use the first 2 atoms of the data set
atoms1 = Atoms('H6C2F', [(sample_df['x coordinates_0'][0], sample_df['y coordinates_0'][0], sample_df['z coordinates_0'][0]), (sample_df['x coordinates_1'][0], sample_df['y coordinates_1'][0], sample_df['z coordinates_1'][0]),(sample_df['x coordinates_2'][0], sample_df['y coordinates_2'][0], sample_df['z coordinates_2'][0]),(sample_df['x coordinates_3'][0], sample_df['y coordinates_3'][0], sample_df['z coordinates_3'][0]),(sample_df['x coordinates_4'][0], sample_df['y coordinates_4'][0], sample_df['z coordinates_4'][0]),(sample_df['x coordinates_5'][0], sample_df['y coordinates_5'][0], sample_df['z coordinates_5'][0]),(sample_df['x coordinates_6'][0], sample_df['y coordinates_6'][0], sample_df['z coordinates_6'][0]),(sample_df['x coordinates_7'][0], sample_df['y coordinates_7'][0], sample_df['z coordinates_7'][0]),(sample_df['x coordinates_8'][0], sample_df['y coordinates_8'][0], sample_df['z coordinates_8'][0])])

atoms2 = Atoms('FH6C', [(sample_df['x coordinates_0'][1], sample_df['y coordinates_0'][1], sample_df['z coordinates_0'][1]), (sample_df['x coordinates_1'][1], sample_df['y coordinates_1'][1], sample_df['z coordinates_1'][1]),(sample_df['x coordinates_2'][1], sample_df['y coordinates_2'][1], sample_df['z coordinates_2'][1]),(sample_df['x coordinates_3'][1], sample_df['y coordinates_3'][1], sample_df['z coordinates_3'][1]),(sample_df['x coordinates_4'][1], sample_df['y coordinates_4'][1], sample_df['z coordinates_4'][1]),(sample_df['x coordinates_5'][1], sample_df['y coordinates_5'][1], sample_df['z coordinates_5'][1]),(sample_df['x coordinates_6'][1], sample_df['y coordinates_6'][1], sample_df['z coordinates_6'][1]),(sample_df['x coordinates_7'][1], sample_df['y coordinates_7'][1], sample_df['z coordinates_7'][1])])



#need a list of the atoms
soap_desc = SOAP(species=["C", "H", "F",  "Cl", "N", "Br"],rcut=5, nmax=8, lmax=6) #, crossover=True)


#create soap descriptor
soap1 = soap_desc.create(atoms1) #, positions=[0])
soap2 = soap_desc.create(atoms2) 

