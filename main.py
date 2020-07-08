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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import statistics as st
import pandas as pd
# import matplotlib.pyplot as plt
# matplotlib inline
# torch.manual_seed(2)

# Função que guarda os resultados em arquivo
def saveExperiments(epoc,addr):    
    arq = open(addr, 'a')
    text = str(epoc) + "\n"
    arq.write(text)
    arq.close()
    

class XOR(nn.Module):
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x
    
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

model = XOR()

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

steps = X.size(0)

# PARAMETROS DE CONFIGURAÇÃO DOS EXPERIMENTOS

# Limite de épocas necessárias sem mudança no erro
# para que seja considerada a estagnação da rede
stagnation = 1000

# Limite máximo de épocas para cada experimento
epochLimit = 5000

# Número de experimentos
experiments = 10

# Variável que conta o número de épocas
epochs = 0

# Lista com um conjunto de taxas de aprendizado 
LRate = [0.1,0.01,0.001,0.0001]

# Estruturas para guaradar o número de épocas por experimento
# e gerar o arquivo de saída em Tex
numbers = []
table = []

# Endereço para guardar os arquivos gerados
address = '/home/serafim/git/pytorchExperiments/experimento_' + str(1)

# TODO - AINDA É NECESSÁRIO ESTUDAR A BIBLIOTECA A RESPEITO DOS OTIMIZADORES
# POR ELA UTILIZADOS. OUTRA INFORMAÇÃO QUE PRECISO ENTENDER É SOBRE A TAXA
# DE APRENDIZADO, OU SEJA, ONDE ELA ENTRA COMO PARÂMETRO. INICIALMENTE ME
# PREOCUPEI EM ESCREVER O CODIGO PARA ATENDER ALGUMAS DOS NOSSOS PARAMÉTROS
# DOS EXPERIMENTOS. PARA SOMENTE DEPOIS ESTUDAR A FUNDO ESTES OUTROS ASPECTOS
# DA BIBLIOTECA.

# Aqui teremos uma quantidade LRate de experimentos.
# Para cada taxa vamos fazer um número experiments de vezes

# TODO - QUANDO A QUESTÃO DA TAXA FOR DEVIDAMENTE ESCLARECIDA ESTA INFORMAÇÃO
# LRate SERÁ UTILIZADA PARA SER INSERIDA NO DEVIDO LOCAL.
for rate in LRate:
    
    # Experimentos por LRate
    for i in range(experiments):
        
        # Inicialização dos pesos.
        weights_init(model)
        # Função de erro.
        loss_func = nn.MSELoss()
        # Otimizador.
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        
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
            # Se atingir o limite de experimentos o experimento para
            if countError == stagnation or epochs == 10000:
                epochs = 0
                break
                
            # Atribui o erro da epocaq anterior
            previousError = currentError
                
            epochs += 1
               
            # Salva o experimento se o erro é zero, ou seja,
            # se acertou os quatro padões.
            if loss.data.numpy() == 0.0:            
                saveExperiments(epochs,address);
                numbers.append(epochs)
                epochs = 0
                break   
        
    # Preenche a tabela com as estatísticas    
    data = {'Learning Rate':rate,'Min':np.min(numbers),'Mean':st.mean(numbers),'Max':np.max(numbers),'Standart Deviation':st.pstdev(numbers)}
    table.append(data)
    
# Salva em arquivo .tex
df = pd.DataFrame(table)
string = df.to_latex(index=False)

arq = open('estatistics.tex', 'a')
text = string
arq.write(text)
arq.close()    

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

