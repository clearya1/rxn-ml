# rxn-ml

We investigate and implement a number of deep learning models to predict quantities of interest in chemical reactions. In particular, we use the QMrxn20 dataset which consists of 7500 reactant molecules which have been constructed from ethane and various substituents. This work proposes 3 models, which classify molecules as transition states, predict the activation energy for that state, and generate the interatomic distances of the transition state. We demonstrate that statistical learning models such as a random forest and linear regression perform sufficiently well on the classification and energy prediction problems, and we make first steps towards the more complex problem of geometry prediction of transition states.

## Getting Started

These instructions will get you a copy of the project up and running on your local or remote machine for development and testing purposes.

### Prerequisites

1. PyTorch
2. DScribe 
3. scikit-learn

## Workflow for Classification of Transition States

1. data\_processing.py
2. prep\_classifier\_data.py
3. mlpClassifier.py

## Workflow for Activation Energy Multilayer Perceptron Regression

1. make\_soap.py
2. data\_sorter.py
3. MLP\_energy\_regressor.py

## Workflow for Interatomic Distances Multilayer Perceptron Regression

1. distancebond.py
2. make\_bond\_files.py
3. split\_data.py
4. MLP\_distance\_regressor.py or GPU\_MLP\_distance\_regressor.py

## Authors

Aidan Tully, Andrew Cleary, Maia Trower, Yasmin Hengster

## Acknowledgments

The authors would like to thank James McDonagh and Clyde Fare from IBM, and Ben Goddard and Antonia Mey from the University of Edinburgh for their support and direction. 
