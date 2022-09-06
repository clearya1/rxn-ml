#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:14:24 2022

@author: s2122199
"""


import numpy as np
from pandas import read_csv

import os

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from os import listdir
import matplotlib.colors as mcolors

folder_path = '/home/aidan/Documents/PhD/Year2/ibm_project/data_files/qmrxn/mlp_outs_low_epoch/' # '_test/'
folders = [folder_path+temp for temp in listdir(folder_path)]
names = listdir(folder_path)

plt.rcParams['font.size'] = '26'
# plt.rcParams['figure.figsize'] = (20,12)

   
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
   
fig, ax = plt.subplots(1,2,sharey=False, figsize=(15,10))
 
colormap = plt.cm.gist_rainbow #winter #nipy_spectral, Set1,Paired  
colors = [colormap(i) for i in np.linspace(0, 1,len(names))]  

# for i in range(len(names)):

#     test_loss = np.load(folders[i]+'/test_loss.npy')
#     training_loss = np.load(folders[i]+'/training_loss.npy')
#     validation_loss = np.load(folders[i]+'/validation_loss.npy')
   
   
 
#     ax[0].plot(training_loss, label = names[i], c = colors[i] )
#     ax[0].set_title('training')
#     # ax[0].set_ylim(0,0.35)
#     ax[1].plot(validation_loss, label = names[i], c=colors[i])
#     ax[1].set_title('validation')
   
#     ax[1].legend(ncol = 2 , bbox_to_anchor=(1.1, 1))
   
#     ax[0].set_ylabel('loss')
#     ax[0].set_xlabel('epoch')
#     ax[1].set_xlabel('epoch')
#     #ax[1].legend()
# #############################################
# fig, ax = plt.subplots(figsize=(15,10))
# test_loss = []
# for i in range(len(names)):

#     test_loss.append(np.load(folders[i]+'/test_loss.npy'))
    
 
# plt.bar(range(len(names)), test_loss,align='center')
# plt.xticks(range(len(names)), names, rotation=90)
# ax.set_title('Test')
# ax.set_ylabel('loss')

# # ax[0].set_ylim(0,0.35)
   
# plt.savefig('dis_reg_test_full.png')

# ######################################
   
#ax[2].plot(test_loss, label = 'test')

# fig.subplots_adjust(top=0.917,bottom=0.127,left=0.071,right=0.696,hspace=0.2,wspace=0.079)
##plot number of datapoints

# %%


# folder_path = '/home/aidan/Documents/PhD/Year2/ibm_project/data_files/qmrxn/bonds_files/' # '_test/'
# folders = [folder_path+temp for temp in listdir(folder_path)]
# names = listdir(folder_path)
# fig, ax = plt.subplots(figsize=(15,10))

# num_files = []
# for i in range(len(names)):
#     files = listdir(folders[i])
#     num_files.append(len(files))
    
# plt.bar(range(len(names)), num_files,align='center')
# plt.xticks(range(len(names)), names, rotation=90)
# ax.set_title('Data size')
# ax.set_ylabel('count')
# plt.savefig('data_size.png')


# input_path = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/small_bonds/'
# files = [input_path+temp for temp in listdir(input_path)]
# names = listdir(input_path)
# for i in range(len(names)):
#     names[i] = names[i][:-4]
# length = np.empty(len(names))

# for i in range(len(names)):
#     data = read_csv(files[i])
#     length[i] = len(data)

# x = np.arange(len(length))    
# fig, ax = plt.subplots(figsize = (20,7))
# plt.rcParams['font.size'] = '18'
# plt.plot(x,length, ls = '', marker = 'x')
# plt.xticks(x, list(names))
# plt.ylabel('no of datapoints')
# plt.tight_layout()
# plt.grid()
# plt.show()

