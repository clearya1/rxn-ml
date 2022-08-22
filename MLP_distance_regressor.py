#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:59:03 2022

@author: s1997751
"""

import numpy as np
from pandas import read_csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from os import listdir
from pathlib import Path

# custom dataset definition
class CSVDataset(Dataset):
  # load the dataset
  def __init__(self, path):
    # load the csv file as a dataframe
    df = read_csv(path, header=None)
    # store the inputs and outputs
    self.X = df.values[1:, 2:-1]
    self.y = df.values[1:, -1]
    # ensure input and output data is floats
    self.X = self.X.astype('float32')
    self.y = self.y.astype('float32')
    self.y = self.y.reshape((len(self.y), 1))
    self.label = str(df.values[1,1])

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]
    
class Data_Loaders():
    def __init__(self, path, batch_size, split_prop=0.8):
        self.path_dataset = CSVDataset(path)
        self.label = self.path_dataset.label
        
        # compute number of samples
        self.N_train = int(len(self.path_dataset) * split_prop)
        self.N_validate = int(len(self.path_dataset) * 0.5*(1.-split_prop))
        self.N_test = len(self.path_dataset) - self.N_train - self.N_validate
        
        if self.N_test == 0 or self.N_validate == 0:
            print(f'-----Need more samples for {self.label}!!---------')

        # split the data set
        self.train_set, self.validate_set, self.test_set = random_split(self.path_dataset, \
                                      [self.N_train, self.N_validate, self.N_test])
        
        # create the data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
        self.validate_loader = DataLoader(self.validate_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, n1, n2, n3):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(560, n1),
        nn.ReLU(),
        nn.Linear(n1, n2),
        nn.ReLU(),
        nn.Linear(n2, n3),
        nn.ReLU(),
        nn.Linear(n3, 1),
        nn.ReLU(),
        )
    
    
    def forward(self, x):
        '''
      Forward pass
        '''
        return self.layers(x)
    
        
def test(model, test_loader, loss_function):
    model.eval()
    test_loss = list()

    with torch.no_grad():
      for data in test_loader:
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
            
            
        outputs=model(inputs)
        targets = targets.view(-1,1)
        
        loss=loss_function(outputs,targets)
        test_loss.append(loss.item())
    
    rmse_test_loss=np.sqrt(np.mean(test_loss))
    
    #print('Test Loss: %.3f' %(rmse_test_loss))
    return rmse_test_loss



def train_and_test_model(path):
    
    # setting parameters here as input must only be path to dataset
    epochs = 10
    model = MLP(256, 128, 64)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    batch_size = 8
    
    # create data loaders
    data_loaders = Data_Loaders(path, batch_size)
    trainloader = data_loaders.train_loader
    validateloader = data_loaders.validate_loader
    testloader = data_loaders.test_loader
    label = data_loaders.label
    
    # start training
    model.train()
    running_loss_mean = list()
    running_validate_loss_mean = list()
    for e in tqdm(range(epochs)):
        #print(f'Starting epoch {e+1}')
        train_loss = list()
        for data in trainloader:
        
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data = data.cuda()
                print("Using GPU")
                
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            outputs = model(inputs)
            # Find the Loss
            # print(outputs.shape, targets.shape)
            targets = targets.view(-1,1)
            loss = loss_function(outputs, targets)
            # Calculate gradients 
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss.append(loss.item())
            
        rmse_validate_loss = test(model, validateloader, loss_function)
            
        running_loss_mean.append(np.sqrt(np.mean(train_loss)))
        running_validate_loss_mean.append(rmse_validate_loss)
        #print(f'Epoch {e+1} \t Training Loss: {running_loss_mean[-1]:.2f} \t Validation Loss: {running_validate_loss_mean[-1]:.2f}')

    # test the model
    test_loss = test(model, testloader, loss_function)
    #print('Training process has finished.')
    print(f'Model {label}, Test Loss: {test_loss:.2f}')
    
    return label, running_loss_mean, running_validate_loss_mean, test_loss, model.state_dict()
    
    # save the model and also the training and validation curves and testing loss
    
    # results_path = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/DistanceModels/'+label
    # print(results_path)
    # os.mkdir(results_path)
    
    
    # torch.save(model.state_dict(), results_path+'/model')
    # np.save(results_path+'/training_loss.npy', running_loss_mean)
    # np.save(results_path+'/validation_loss.npy', running_validate_loss_mean)
    # np.save(results_path+'/test_loss.npy', test_loss)
    
    #return running_loss_mean, running_validate_loss_mean, test_loss

    
####%%
if __name__ == '__main__':
    
    # create list of paths to files
    #folder_path = '/Users/Andrew/Documents/Edinburgh/ChemistyML/bond_files1/'
    folder_path = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/small_bonds/'
    paths = [folder_path+temp for temp in listdir(folder_path)]
    names = listdir(folder_path)
  
    with ProcessPoolExecutor(max_workers=4) as executor:
      results = list(tqdm(executor.map(train_and_test_model, paths), total=len(paths)))
    
    for i in results:
        results_path = '/home/s2122199/Documents/Edinburgh/projects/IBM/data/QMrxn/geometries/DistanceModels_difflearning/' + i[0]
        if not Path(results_path).exists():
            Path(results_path).mkdir(parents = True)
        np.save(results_path+'/training_loss.npy', i[1])
        np.save(results_path+'/validation_loss.npy', i[2])
        np.save(results_path+'/test_loss.npy', i[3])
        torch.save(i[4], results_path+'/model')
        
