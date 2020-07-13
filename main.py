#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:31:09 2020

@author: serafim
"""

# Imports
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import statistics as st
import pandas as pd
import os.path
# import matplotlib.pyplot as plt
# matplotlib inline
# torch.manual_seed(2)
    
# Função que guarda ou recupera as époćas.
# 0 para salvar e 1 para recuperar
def saveEpochs(addr,epoc,num):
    arq = open(addr + 'experimento_' + str(num), 'a')
    text = str(epoc) + "\n"
    arq.write(text)
    arq.close()        
    
# Recupera as epocas por taxa e gera as estatísticas.
def recoverEpochs(addr,rates):
    numbers = []
    table = []
    for rate in rates:
        isExist = os.path.exists(addr + 'experimento_' + str(rate))
        if (isExist == True):
            arq = open(addr + 'experimento_' + str(rate), 'r')
            linha = arq.readlines()
            numbers = list(map(int,linha))
            data = {'Learning Rate':rate,'Min':np.min(numbers),'Mean':st.mean(numbers),'Max':np.max(numbers),'Standart Deviation':st.pstdev(numbers)}
            table.append(data)
            arq.close()
    
    df = pd.DataFrame(table)
    string = df.to_latex(index=False)

    arq = open('estatistics.tex', 'a')
    text = string
    arq.write(text)
    arq.close()
    
# Função que guarda o status dos experimentos
def saveStatus(addr,rat,exper):
    arq = open(addr + 'status', 'w+')
    text = str(rat) + "\n" + str(exper)
    arq.write(text)
    arq.close()       
    
# Função que inicializa ou reinicializa o status dos experimentos
def arqExist(addr,rat,expr):
    isExist = os.path.exists(addr + 'status')
    if (isExist == True):
        arq = open(addr + 'status', 'r')
        linha = arq.readlines()
        arq.close()
        rat = linha[0]
        epoc = linha[1]
        return np.float(rat),np.float(epoc)
    else:
        saveStatus(addr,rat,expr)
        return rat,expr
        
# FUNÇÕES PYTORCH
class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        return x
    
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

# PARAMETROS DE CONFIGURAÇÃO DO PYTORCH
model = XOR()

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

steps = X.size(0)

# PARAMETROS DE CONFIGURAÇÃO DOS EXPERIMENTOS

# Limite de épocas necessárias sem mudança no erro
# para que seja considerada a estagnação da rede
stagnation = 500

# Limite máximo de épocas para cada experimento
epochLimit = 1000

# Número de experimentos por taxa
experiments = 1200

# Variável que conta o número de épocas
epochs = 0

# Lista com um conjunto de taxas de aprendizado 
LRate = [0.1,0.01,0.001,0.0001]

# Variáveis auxiliares de inicialização
rate = 0.0
expr = 0

# Endereço para guardar os arquivos gerados
address = '/home/serafim/git/neuralnetworkpytorch/'

# Verifica se o status já existe. Se não existir
# então será dado inicio aos experimentos. Caso o status
# exista é dado continuidade.
rate,expr = arqExist(address,LRate[0],0)

# Aqui teremos uma quantidade LRate de experimentos.
# Para cada taxa vamos fazer um número experiments de vezes
while True:
    
    print("Inicio para taxa: ",rate)
    
    # Se houver reinicialização do programa só
    # será realizado os experimentos se ainda não
    # foram todos feitos.
    if expr == experiments and rate == LRate[len(LRate)-1]:
        break
    
    while expr < experiments:
        
        saveStatus(address,rate,expr)
        
        # Inicialização dos pesos.
        weights_init(model)
        # Função de erro.
        loss_func = nn.MSELoss()
        # Otimizador.
        optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.9)
        
        # Variáveis que contabilizam os erros para
        # estagnação da rede.
        previousError = 1.0
        currentError = 0.0
        countError = 0
        
        # O experimento ocorre enquanto uma solução não for encontrada ou
        # a rede não estagnar ou não chegar no número máximo
        # de épocas.
        while True:
            
            # São apresentados quatro padrões aleatórios.
            # Ao final da apresentação dos quatro padrões
            # contabiliza-se uma época.
            for j in range(steps):
                data_point = np.random.randint(X.size(0))
                x_var = Variable(X[data_point], requires_grad=False)
                y_var = Variable(Y[data_point], requires_grad=False)
                
                optimizer.zero_grad()
                y_hat = model(x_var)
                loss = loss_func.forward(y_hat, y_var)
                loss.backward()
                optimizer.step()
            
            # Atribui o erro atual
            currentError = loss.data.numpy()
            
            # Inicia ou Reinicia a contagem para estagnação
            if currentError == previousError:
                countError += 1
            else:
                countError = 0
                
            # Verifica se a rede estagnou
            # Se atingir o limite de epocas o experimento para
            if countError == stagnation or epochs == epochLimit:
                epochs = 0
                break
                
            # Atribui o erro da epocaq anterior
            previousError = currentError
                
            epochs += 1
               
            # Salva o experimento se o erro é zero, ou seja,
            # se acertou os quatro padões.
            if loss.data.numpy() == 0.0:            
                saveEpochs(address,epochs,rate)
                epochs = 0
                break
        
        # Atualiza o número de experimentos por taxa
        expr += 1
    
    # Verifica se é a última taxa.
    # Se for então é matida a taxa e o
    # número de experimentos caso seja preciso
    # reiniciar.
    if rate != LRate[len(LRate)-1]:
        print("Término para taxa: ",rate)
        rate = LRate[LRate.index(rate)+1]
        expr = 0
    # Salva o status atual do experimento
    saveStatus(address,rate,expr)
    
# Recupera épocas por experimento e gera as estatísticas
recoverEpochs(address,LRate)   

# Plots
'''
model_params = list(model.parameters())

model_weights = model_params[0].data.numpy()
model_bias = model_params[1].data.numpy()

plt.scatter(X.numpy()[[0,-1], 0], X.numpy()[[0, -1], 1], s=50)
plt.scatter(X.numpy()[[1,2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

x_1 = np.arange(-0.1, 1.1, 0.1)
y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
plt.plot(x_1, y_1)

x_2 = np.arange(-0.1, 1.1, 0.1)
y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
plt.plot(x_2, y_2)
plt.legend(["neuron_1", "neuron_2"], loc=8)
plt.show()
'''

