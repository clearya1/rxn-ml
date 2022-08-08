#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:59:03 2022

@author: s1997751
"""

import numpy as np
import pandas as pd
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import IterableDataset
from torchvision.datasets import DatasetFolder
import torch
import os, io

#path to .soap files
path = '/home/s1997751/Documents/PhD/Year2/ibm_project/data_files/qmrxn/'
# file with tuples
master_file = '/home/s1997751/Documents/PhD/Year2/ibm_project/data_files/qmrxn/tuple_list_energy.npy'


# custom dataset definition

class PathDataset(Dataset):

    def __init__(self, folder_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with types of directories.
            transform (callable, optional): Optional transform to be applied
                on a sample. -> could implement soap here if needed
        """
        self.files = np.load(folder_file)
        self.files = self.files[:10000]
        #print(self.files)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        reactant = np.loadtxt(str(self.root_dir) + str(self.files[idx][0]))
   
        product = np.loadtxt(str(self.root_dir) + str(self.files[idx][1]))
        ##to predict energy:
        transformer = self.files[idx][2]
        ##to predict soap representation
        # transformer = np.loadtxt(str(self.root_dir) + str(self.files[idx][2]))

        sample = {'coordinates': np.array([reactant,product], dtype=(float)), 'energy': np.array(transformer, dtype=(float))}

        #if self.transform:
            #sample = self.transform(sample)
       
        return sample
    
class Data_Loaders():
    def __init__(self, master_file, path, batch_size, split_prop=0.8):
        self.path_dataset = PathDataset(master_file, path)
        # compute number of samples
        self.N_train = int(len(self.path_dataset) * 0.8)
        self.N_test = len(self.path_dataset) - self.N_train

        self.train_set, self.test_set = random_split(self.path_dataset, \
                                      [self.N_train, self.N_test])
        
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)

   
# batch_size = 16
# data_loaders = Data_Loaders(master_file, path, batch_size)

# train_features= next(iter(data_loaders.train_loader))
# print(train_features['coordinates'].shape)



class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(56, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
        )
    
    
    def forward(self, x):
        '''
      Forward pass
        '''
        return self.layers(x)
    
      
if __name__ == '__main__':
  
    # Set fixed random number seed
    torch.manual_seed(42)
    
    batch_size = 16
    data_loaders = Data_Loaders(master_file, path, batch_size)
    
    trainloader = data_loaders.train_loader
    testloader = data_loaders.test_loader
    
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch+1}')
    
        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
      
            # Get and prepare inputs
            inputs, targets = data['coordinates'], data['energy']
            # inputs, targets = inputs.float(), targets.float()
            inputs = inputs.float()
            # targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

# Process is complete.
print('Training process has finished.')
  
#save model
# torch.save(mlp.state_dict(), './energy_regressor.pt')
#load model
# model = MLP()
# model.load_state_dict(torch.load('./energy_regressor.pt'))
  
with torch.no_grad():
    mlp.eval()
    # test_inputs, test_targets = testloader['coordinates'], testloader['energy']
      # inputs, targets = inputs.float(), targets.float()
    # test_inputs = test_inputs.float()
    # y_pred = mlp(test_inputs)
    # test_loss = criterion(y_pred, test_targets)

