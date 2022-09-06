# pytorch mlp for binary classification
import numpy as np
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt

# custom dataset definition
class CSVDataset(Dataset):
  # load the dataset
  def __init__(self, path, y_label):
    # load the csv file as a dataframe
    df = read_csv(path, header=None)
    # store the inputs and outputs
    #self.X = df.drop(y_label, axis=1)
    self.X = df.values[:, :-1]
    #self.y = df[y_label].values
    self.y = df.values[:, -1]
    # ensure input data is floats
    self.X = self.X.astype('float32')
    # label encode target and ensure the values are floats
    self.y = LabelEncoder().fit_transform(self.y)
    self.y = self.y.astype('float32')
    self.y = self.y.reshape((len(self.y), 1))

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]

  # get indexes for train and test rows
  def get_splits(self, n_test=0.2):
    # determine sizes
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size
    # calculate the split
    return random_split(self, [train_size, test_size])
    
# custom dataset definition
class NumpyDataset(Dataset):
  # load the dataset
  def __init__(self, path, y_label):
    # load the csv file as a dataframe
    nparray = np.load(path, allow_pickle=True)
    rows = len(nparray)
    cols = len(nparray[0][0])
    # store the inputs and outputs
    self.X = np.zeros((rows, cols))
    self.y = np.zeros(rows)
    for i in range(rows):
      self.X[i] = nparray[i][0]
      self.y[i] = nparray[i][1]
    # ensure input data is floats
    self.X = self.X.astype('float32')
    # label encode target and ensure the values are floats
    self.y = LabelEncoder().fit_transform(self.y)
    self.y = self.y.astype('float32')
    self.y = self.y.reshape((len(self.y), 1))
    print(int(sum(self.y)[0]), ' transition states out of ', rows)

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]

  # get indexes for train and test rows
  def get_splits(self, n_test=0.2):
    # determine sizes
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size
    # calculate the split
    return random_split(self, [train_size, test_size])

# Define model
class MLP(Module):
  # define model elements
  def __init__(self, n_inputs):
    super(MLP, self).__init__()
    # input to first hidden layer
    self.hidden1 = Linear(n_inputs, 64)
    kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
    self.act1 = ReLU()
    # second hidden layer
    #self.hidden2 = Linear(64, 64)
    #kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
    #self.act2 = ReLU()
    # third hidden layer and output
    self.hidden3 = Linear(64, 1)
    xavier_uniform_(self.hidden3.weight)
    self.act3 = Sigmoid()

  # forward propagate input
  def forward(self, X):
    # input to first hidden layer
    X = self.hidden1(X)
    X = self.act1(X)
     # second hidden layer
    #X = self.hidden2(X)
    #X = self.act2(X)
    # third hidden layer and output
    X = self.hidden3(X)
    X = self.act3(X)
    return X
    
# prepare the dataset
def prepare_data(path, y_label):
  # load the csv file as a dataframe
  dataset = NumpyDataset(path, y_label)
  # calculate split
  train, test = dataset.get_splits()
  # prepare data loaders
  train_dl = DataLoader(train, batch_size=4098, shuffle=True)
  test_dl = DataLoader(test, batch_size=4098, shuffle=False)
  return train_dl, test_dl
  
# train the model
def train_model(train_dl, model):
  # define the optimization
  criterion = BCELoss()
  optimizer = Adam(model.parameters(), lr=0.01)
  running_loss_mean = list()
  # enumerate epochs
  for epoch in range(150):
    # enumerate mini batches
    print('Epoch ', epoch, '/150')
    train_loss = list()
    for i, (inputs, targets) in enumerate(train_dl):
      # clear the gradients
      optimizer.zero_grad()
      # compute the model output
      yhat = model(inputs)
      # calculate loss
      loss = criterion(yhat, targets)
      # credit assignment
      loss.backward()
      # update model weights
      optimizer.step()
      # save loss from minibatch
      train_loss.append(loss.item())
    
    epochloss = np.mean(np.array(train_loss))
    running_loss_mean.append(epochloss)
    print('Training Loss = ', epochloss)
    
  return running_loss_mean
      
# evaluate the model
def evaluate_model(test_dl, model):
  predictions, actuals = list(), list()
  for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat = model(inputs)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    actual = targets.numpy()
    actual = actual.reshape((len(actual), 1))
    # round to class values
    yhat = yhat.round()
    # store
    predictions.append(yhat)
    actuals.append(actual)
  predictions, actuals = vstack(predictions), vstack(actuals)
  # calculate accuracy
  acc = accuracy_score(actuals, predictions)
  return acc

# make a class prediction for one row of data
def predict(row, model):
  # convert row to data
  row = Tensor([row])
  # make prediction
  yhat = model(row)
  # retrieve numpy array
  yhat = yhat.detach().numpy()
  return yhat
  
# prepare the data
#path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
path = '/Users/Andrew/Documents/Edinburgh/ChemistyML/soap_transition_n1l0.npy'
train_dl, test_dl = prepare_data(path, -1)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
n_inputs = 28
model = MLP(n_inputs)
# train the model
trainingloss = train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

print(trainingloss)

plt.plot(np.arange(1,151,1), trainingloss)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.show()
        
  
