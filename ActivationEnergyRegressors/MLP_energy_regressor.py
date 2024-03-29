import numpy as np
import pandas as pd
from numpy import vstack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


#path to .soap files, created by make_soap.py
path = '/home/acleary/data_files/qmrxn/'
# file with tuples (path to reactant complex, path to product conformer, transition state energy) created by data_sorter.py
master_file = '/home/acleary/data_files/qmrxn/tuple_list_energy.npy'


# custom dataset definition

class PathDataset(Dataset):

    def __init__(self, folder_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with types of directories.
        """
        self.files = np.load(folder_file)
        #print(self.files)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        reactant = np.loadtxt(str(self.root_dir) + str(self.files[idx][0]))
   
        product = np.loadtxt(str(self.root_dir) + str(self.files[idx][1]))
        ##to predict energy:
        transformer = self.files[idx][2]

        #sample = {'coordinates': np.array(np.hstack((reactant,product)), dtype=(float)), 'energy': np.array(transformer, dtype=(float))}

        sample = {'coordinates': np.array(reactant, dtype=(float)), 'energy': np.array(transformer, dtype=(float))}
       
        return sample
    
class Data_Loaders():
    def __init__(self, master_file, path, batch_size, split_prop=0.8):
        self.path_dataset = PathDataset(master_file, path)
        # compute number of samples
        self.N_train = int(len(self.path_dataset) * 0.8)
        self.N_validate = int(len(self.path_dataset) * 0.1)
        self.N_test = len(self.path_dataset) - self.N_train - self.N_validate

        self.train_set, self.validate_set, self.test_set = random_split(self.path_dataset, \
                                      [self.N_train, self.N_validate, self.N_test])
        
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
        self.validate_loader = DataLoader(self.validate_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, n1, n2):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(56, n1),
        nn.ReLU(),
        nn.Linear(n1, n2),
        nn.ReLU(),
        nn.Linear(n2, 1)
        )
    
    
    def forward(self, x):
        '''
      Forward pass
        '''
        return self.layers(x)
        
# define our custom loss function
def rmsre(outputs, targets):
    '''
    Root mean squared relative error
    '''

    return torch.sqrt(torch.mean(torch.pow(torch.div(torch.sub(outputs,targets),targets),2)))
    
        
def test(model, test_loader, loss_function):
    model.eval()
    test_loss = list()

    with torch.no_grad():
      for data in tqdm(test_loader):
        inputs, targets = data['coordinates'], data['energy']
        inputs, targets = inputs.float(), targets.float()
            
            
        outputs=model(inputs)
        targets = targets.view(-1,1)
        
        loss=loss_function(outputs,targets)
        test_loss.append(loss.item())
    
    rmse_test_loss=np.sqrt(np.mean(np.array(test_loss)**2))
    
    return rmse_test_loss



def train_model(model, trainloader, validateloader, loss_function, optimizer):
    
    epochs = 150
    model.train()
    running_loss_mean = list()
    running_validate_loss_mean = list()
    for e in range(epochs):
        print(f'Starting epoch {e+1}')
        train_loss = list()
        for data in tqdm(trainloader):
            inputs, targets = data['coordinates'], data['energy']
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
            
        # running_loss.append(train_loss)
        running_loss_mean.append(np.sqrt(np.mean(np.array(train_loss)**2)))
        running_validate_loss_mean.append(np.array(rmse_validate_loss)**2)
        print(f'Epoch {e+1} \t Training Loss: {running_loss_mean[-1]:.5f} \t Validation Loss: {running_validate_loss_mean[-1]:.5f}')

    print('Training process has finished.')
    return running_loss_mean, running_validate_loss_mean

    
if __name__ == '__main__':
    
    batch_size = 16
    data_loaders = Data_Loaders(master_file, path, batch_size)
        
    trainloader = data_loaders.train_loader
    validateloader = data_loaders.validate_loader
    testloader = data_loaders.test_loader
    
    # Initialize the MLP
    n1 = 64
    n2 = 32
    mlp = MLP(n1, n2)
    
    # Define the loss function and optimizer
    loss_function = rmsre
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    
    loss_out, validate_loss_out = train_model(mlp, trainloader, validateloader, loss_function, optimizer)
    
    os.mkdir('TrainingResultsNoProd/'+str(n1)+'_'+str(n2)+'_batchsize'+str(batch_size))
    np.save('TrainingResultsNoProd/'+str(n1)+'_'+str(n2)+'_batchsize'+str(batch_size)+'/training_loss.npy', loss_out)
    np.save('TrainingResultsNoProd/'+str(n1)+'_'+str(n2)+'_batchsize'+str(batch_size)+'/validation_loss.npy', validate_loss_out)
    
    test_loss = test(mlp, testloader,loss_function)
    print('Test Loss = ', test_loss)
