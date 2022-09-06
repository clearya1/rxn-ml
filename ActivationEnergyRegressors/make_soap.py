from tqdm import tqdm
import numpy as np
import pandas as pd 
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix
from ase import Atoms
from pathlib import Path

#make a list of all coordinate files
IN_PATH = Path('/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/')
files = list( IN_PATH.glob("**/*.xyz") )
print('files to choose from: ', len(files))

#parameters for soap funtion
species=["C", "H", "F",  "Cl", "N", "Br", "O"]
rcut = 7
nmax = 1
lmax = 1

#function to create soap array, average gives same shape, no matter how many atoms a molecule contains
av_soap = SOAP(species=species, rcut=rcut, nmax=nmax, lmax=lmax, average="inner", sparse=False)

#iterate over the .xyz files, create molecule name and list of positions to feed into the soap function, save each soap array as .soap 
for file in tqdm(files,total=len(files)):
    coords = pd.read_csv(file, skiprows = 2, delim_whitespace = True, header = None)
    mol_name = ''.join(coords[0])
    pos = np.array(coords[[1,2,3]])
    atom = Atoms(mol_name,pos)
    des = av_soap.create(atom)
    np.savetxt(str(file)[:-3]+'soap',des)
    
    
