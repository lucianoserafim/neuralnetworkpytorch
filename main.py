#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:31:09 2020

This code performs the training of a neural network
to solve the XOR problem using the pythorch library.

@author: Serafim, Jonathan and Tiago
"""

# IMPORTS
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import statistics as st
import pandas as pd
import os.path
import random
# import matplotlib.pyplot as plt
# matplotlib inline
# torch.manual_seed(2)
    
# Function to save the status of epochs.
def saveEpochs(addr,epoc,rate,el,stag,expr):
    arq = open(addr + 'experiment_rate_' + str(rate) + '_el_' + str(el) + '_stag_' + str(stag) + '_exper_' + str(expr), 'a')
    text = str(epoc) + "\n"
    arq.write(text)
    arq.close()        
    
# Retrieves the epochs by rate and generates statistics.
def recoverEpochs(addr,rates,el,stag,expr):
    numbers = []
    table = []
    for rate in rates:
        pathAddr = addr + 'experiment_rate_' + str(rate) + '_el_' + str(el) + '_stag_' + str(stag) + '_exper_' + str(expr)
        isExist = os.path.exists(pathAddr)
        if (isExist == True):
            arq = open(pathAddr, 'r')
            linha = arq.readlines()
            numbers = list(map(int,linha))
            data = {'Learning Rate':rate,'Min':np.min(numbers),'Mean':st.mean(numbers),'Max':np.max(numbers),'Standart Deviation':st.pstdev(numbers)}
            table.append(data)
            arq.close()
    
    df = pd.DataFrame(table)
    string = df.to_latex(index=False)

    arq = open(addr + 'estatistics.tex', 'a')
    text = string
    arq.write(text)
    arq.close()
    
# Function that stores the status of experiments.
def saveStatus(addr,rat,exper):
    arq = open(addr + 'status_' + str(rat), 'w+')
    text = str(rat) + "\n" + str(exper)
    arq.write(text)
    arq.close()
    
# Function that stores the number of stagnant epochs or at maximum limit.
def saveError(addr,rat,tp,value):
    arq = open(addr + tp + '_' + str(rat), 'w+')
    text = str(value)
    arq.write(text)
    arq.close()

# Function that retrieves the number of stagnant epochs or at maximum limit.    
def recoverError(addr,rat,tp,value):
    isExist = os.path.exists(addr + tp + '_' + str(rat))
    if (isExist == True):
        arq = open(addr + tp + '_' + str(rat), 'r')
        linha = arq.readline()
        arq.close()
        
        number = np.float(linha)
        soma = value + number
        text = str(soma)
        
        arq = open(addr + tp + '_' + str(rat), 'w')
        arq.write(text)
        arq.close()
    else:
        saveError(addr,rat,tp,value)
        
# Function that initializes or retrieves the status of experiments.
def arqExist(addr,lrat,expr):
    
    for i in lrat:
        isExist = os.path.exists(addr + 'status_' + str(i))
        if (isExist == True):
            arq = open(addr + 'status_' + str(i), 'r')
            linha = arq.readlines()
            arq.close()
            rat = np.float(linha[0])
            numExpr = np.float(linha[1])
            if (numExpr < expr) or (numExpr == expr and i == lrat[len(LRate)-1]):
                return rat,numExpr
        else:
            saveStatus(addr,i,0)
            return i,0
        
# PYTORCH FUNCTIONS
class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        #x = torch.sigmoid(x)
        return x
    
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

# PYTORCH CONFIGURATION PARAMETERS
model = XOR()

X = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

steps = X.size(0)

data_point = [0,1,2,3]

# EXPERIMENT CONFIGURATION PARAMETERS

# Limit of necessary epochs without changing the error to
# consider the stagnation of the network.
stagnation = 1000

# Maximum epochs limit for each experiment.
epochLimit = 150000

# Number of experiments per rate.
experiments = 1200

# Epoch counter.
epochs = 1

# List of learning rates. 
LRate = [0.1,0.01,0.001,0.0001,0.005]

# Auxiliary initialization variables.
rate = 0.0
expr = 0

# Address to save the generated files.
address = 'path to files'

# Checks whether the status already exists. If it does not exist
# then experiments will be started. If the status exists, continuity is given.
rate,expr = arqExist(address,LRate,experiments)

# While not reaching all experiments by the rate of learning.
while True:
    
    # Checks if all experiments are complete.
    if expr == experiments and rate == LRate[len(LRate)-1]:
        break
    
    while expr < experiments:
        
        saveStatus(address,rate,expr)
        
        # Weights initialization.
        weights_init(model)
        # Error function.
        loss_func = nn.MSELoss()
        # Optimizer.
        optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.9)
        
        # Variables that account for successes and errors for network stagnation.
        previousError = 1.0
        currentError = 0.0
        countError = 0
        
        # The experiment occurs as long as a solution is not found
        #or the network does not stagnate or reach the maximum number of epochs.
        while True:
            
            countHit = 0
            
            random.shuffle(data_point)
            
            for j in range(steps):
                
                x_var = Variable(X[data_point[j]], requires_grad=False)
                y_var = Variable(Y[data_point[j]], requires_grad=False)
                
                optimizer.zero_grad()
                y_hat = model(x_var)
                loss = loss_func.forward(y_hat, y_var)
                loss.backward()
                optimizer.step()
                
                if (y_hat > 0.5 and y_var == 1) or (y_hat < 0.5 and y_var == 0):
                    countHit += 1
            
            # Assigns the current error.
            currentError = loss.data.numpy()
            
            if countHit == 4:
                saveEpochs(address,epochs,rate,epochLimit,stagnation,experiments)
                epochs = 1
                break
            
            # Starts or restarts the count for stagnation.
            if currentError == previousError:
                countError += 1
            else:
                countError = 0
                
            # Checks whether the network has stagnated.
            if countError == stagnation:
                # Salva a quantidade de experimentos estagnados
                recoverError(address,rate,'stagnation',1)
                epochs = 1
                break
            
            # The experiment will be stopped if the total number of epochs is reached.
            if epochs == epochLimit:
                # Salva a quantidade de experimentos  no limite de épocas
                recoverError(address,rate,'epochsLimit',1)
                epochs = 1
                break
                
            # Attributes the error of the previous epoch.
            previousError = currentError
                
            epochs += 1
        
        # Updates the number of experiments per rate.
        expr += 1
        print(expr)
        
        # Saves the current status of the experiment.
        saveStatus(address,rate,expr)
    
    # Check if it is the last rate.
    if rate != LRate[len(LRate)-1]:
        rate = LRate[LRate.index(rate)+1]
        expr = 0
    
# Generates the statistics.
recoverEpochs(address,LRate,epochLimit,stagnation,experiments)   

