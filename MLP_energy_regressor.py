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
import torch.nn.functional as F
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
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        self.files = self.files#[:10000]
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

        sample = {'coordinates': np.array(np.hstack((reactant,product)), dtype=(float)), 'energy': np.array(transformer, dtype=(float))}

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




class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(112, 64),
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
    
        
def test(model, test_loader, loss_function):
    model.eval()
    running_loss=0
    total=0

    with torch.no_grad():
      for data in tqdm(test_loader):
        inputs, targets = data['coordinates'], data['energy']
        inputs, targets = inputs.float(), targets.float()
            
            
        outputs=model(inputs)
        targets = targets.view(-1,1)
        
        loss=torch.sqrt(loss_function(outputs,targets))
        running_loss+=loss.item()
    
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
    
    test_loss=running_loss/len(test_loader)
    # accu=100.*correct/total
    
    eval_losses.append(test_loss)
    # eval_accu.append(accu)
    
    print('Test Loss: %.3f' %(test_loss))



def train_model(model, trainloader, loss_function, optimizer):
    
    epochs = 50
    model.train()
    running_loss_mean = list()
    for e in range(epochs):
        print(f'Starting epoch {e+1}')
        train_loss = list()
        for data in tqdm(trainloader):
            inputs, targets = data['coordinates'], data['energy']
            inputs, targets = inputs.float(), targets.float()
            
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data = data.cuda()
    
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            outputs = model(inputs)
            # Find the Loss
            # print(outputs.shape, targets.shape)
            targets = targets.view(-1,1)
            loss = torch.sqrt(loss_function(outputs, targets))
            # Calculate gradients 
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss.append(loss.item())
            
        # running_loss.append(train_loss)
        running_loss_mean.append(np.mean(train_loss))
        print(f'Epoch {e+1} \t\t Training Loss: {torch.tensor(train_loss).mean():.2f}')
    # plt.plot(np.ravel(running_loss))
    plt.plot(running_loss_mean)
    print('Training process has finished.')
    return running_loss_mean

    
#%%
if __name__ == '__main__':
    
    # batch_size = 16
    # data_loaders = Data_Loaders(master_file, path, batch_size)
    
    # train_features= next(iter(data_loaders.train_loader))
    # print(train_features['coordinates'].shape)
    # print(train_features['energy'].shape)
  
    # Set fixed random number seed
    # torch.manual_seed(42)
    
    batch_size = 16
    data_loaders = Data_Loaders(master_file, path, batch_size)
        
    trainloader = data_loaders.train_loader
    testloader = data_loaders.test_loader
    
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    # loss_function = nn.L1Loss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    loss_out = train_model(mlp, trainloader, loss_function, optimizer)
  
   
# Process is complete.

  #%%
#save model
torch.save(mlp.state_dict(), '/home/s1997751/Documents/PhD/Year2/ibm_project/temp/local_code/energy_regressor_1.pt')
#load model
# model = MLP()
# model.load_state_dict(torch.load('/home/s1997751/Documents/PhD/Year2/ibm_project/temp/local_code/energy_regressor.pt'))

#%%
# model = MLP()
# test_inputs, test_targets = data['coordinates'], data['energy']
eval_accu, eval_losses  = list(), list()
test(mlp, testloader,loss_function)

#%%
plt.loglog(loss_out)