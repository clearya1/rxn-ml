# Creates a csv file containing the coordinates of each molecule and its corresponding energy.

import pandas as pd
import numpy as np

path = '/home/s2113337/MAC-MIGS/IBM/'
energies = pd.read_csv(path+'energies.txt')

# get coordinates of all atoms i
#option: make 3 columns with tuples i.e. x col contains all x coordinates etc ??

def get_coordinates(n, energies, cols, all_cols):
    # for row n in energies file ...
    coord_path = path + energies['filename'][n]
    # read coordinates
    coords = pd.read_csv(coord_path, skiprows = 1, delim_whitespace = True)
    # make temporary df with coordinates for each atom
    coord_df = pd.DataFrame(data = coords.values, columns = cols)
    # if there are <17 atoms in compound then pack left over columns with nan
    buffer = np.empty((1,4*(21 - len(coord_df.index)))).reshape(1,-1)
    buffer[:] = np.nan
    data = np.concatenate((coord_df.values.reshape((1,-1)), buffer), axis = 1)
    # make df containing all coordinates 
    df = pd.DataFrame(data, columns = all_cols)
    df.drop(columns='filename',inplace=True)
    return df, all_cols

cols = ['element','x coordinates', 'y coordinates', 'z coordinates'] 
# make enough columns to store all coordinates 
all_cols = [x+'_'+str(i) for i in range(30) for x in cols]
    
# get coordinates for all 
data = [get_coordinates(i, energies, cols, all_cols)[0] for i in range(len(energies.index))]
coord_df = pd.concat(data).reset_index()

coord_df = coord_df.drop(columns = ['index'])

energies_new = pd.concat([energies, coord_df], axis=1)
energies_new.to_csv(path+'energies_coordinates.csv')
