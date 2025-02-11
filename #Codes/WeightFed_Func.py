#%%
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import grad
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import copy
import Common_Func
from torchdiffeq1 import odeint
device = torch.device("cuda:2")

#%%

class NeuralODE_per_newlayer(nn.Module):
    def __init__(self,input_num,output_num,hidden_units,layer):
        super(NeuralODE_per_newlayer, self).__init__()
        self.hidden_units = hidden_units
        self.input_num = input_num
        self.output_num = output_num

        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units))
        layers.append(nn.ReLU())
        for _ in range(layer-1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units, self.output_num))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        comp2 = self.net(z)
        return comp2

class Federated_phy_newlayer(nn.Module):
    def __init__(self, input_num, output_num, hidden_units,layer):
        super(Federated_phy_newlayer, self).__init__()
        self.input_num, self.output_num, self.hidden_units = input_num, output_num, hidden_units
        
        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units))
        layers.append(nn.ReLU())
        for _ in range(layer-1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units, self.output_num))
        self.net = nn.Sequential(*layers)
    
    def forward(self,u):
        comp_out = self.net(u)
        return comp_out

class NeuralODE_merge(nn.Module):
    def __init__(self, nn_per, nn_phy):
        super(NeuralODE_merge, self).__init__()
        self.a = nn.Parameter(torch.tensor(-0.2), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-0.3), requires_grad=True)
        self.nn_per = nn_per
        self.nn_phy = nn_phy

    def forward(self,z):
        comp_fun_1 = self.nn_per(z)
        comp2 = self.nn_phy(comp_fun_1)
        return comp2

class NeuralODE_merge_final(nn.Module):
    def __init__(self, input_num, output_num, nn_merge):
        super(NeuralODE_merge_final, self).__init__()
        self.nn_merge = nn_merge
        self.input_num,self.output_num = input_num,output_num
        self.prepross = nn.Linear(self.input_num, self.output_num)

    def forward(self,t,z):
        y1, y2 = z[:,:,0:1].requires_grad_(True), z[:,:,1:2].requires_grad_(True)
        comp1 = self.nn_merge.a * y1 + self.nn_merge.b * y2
        x = []
        for i in range(self.input_num):
            x.append(z[:,:,(i+2):(i+3)].requires_grad_(True))
        
        comp2_pre = self.prepross(torch.cat([x[i] for i in range(self.input_num)], dim=-1))
        comp2 = self.nn_merge(comp2_pre)
        return comp1 + comp2

class NeuralODE_weighted_2(nn.Module):
    def __init__(self, input_num, output_num,preprocess,merge2,merge3):
        super(NeuralODE_weighted_2, self).__init__()
        self.merge2,self.merge3 = merge2,merge3
        self.prepross_n = nn.Linear(input_num, output_num)
        self.prepross_n.load_state_dict(preprocess)
        self.w_model = torch.nn.Parameter(torch.tensor(0.32), requires_grad=True)
        
    def forward(self,t,z):
        y1, y2 = z[:,:,0:1], z[:,:,1:2]
        comp1 = self.w_model*self.merge2.a*y1+(1-self.w_model)*self.merge3.a*y1\
             + self.w_model*self.merge2.b*y2+(1-self.w_model)*self.merge3.b*y2
        
        comp2_pre = self.prepross_n(z[:,:,2:])
        comp2 = self.w_model*self.merge2(comp2_pre)+(1-self.w_model)*self.merge3(comp2_pre)

        relu = nn.ReLU()
        comp_pen = relu(-1*self.w_model)**2+relu(self.w_model-1)**2

        return comp1 + comp2, comp_pen
