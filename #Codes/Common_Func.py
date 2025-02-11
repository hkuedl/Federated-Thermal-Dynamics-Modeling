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
device = torch.device("cuda:2")
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import copy

seed_value = 20
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

def reorder(X_tr, T_Fre):
    day = int((X_tr.shape[0]-1)/(T_Fre*24))
    for i in range(day-1):
        index = (i+1)*T_Fre*24+i
        X_tr = np.insert(X_tr, index, X_tr[index,:], axis=0)
    return X_tr

def data_input_less(T_Fre,Train_s,Train_e,Train_s2,Train_e2,i_build,building_name,csv_):
    Building_excel = pd.ExcelFile('Data/#Introduction.xlsx').parse('Sheet1')
    Building_zone = Building_excel['zones'].tolist()
    N_zone = Building_zone[i_build]
    T_len = 8760*T_Fre
    AUX_sum = 5*N_zone+1
    if csv_ == 'yes':
        data_in = np.loadtxt('Data/Data_in/'+building_name+'.csv',delimiter=",",skiprows=1,usecols=range(1,AUX_sum+1))
    P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
    T_rad0 = data_in[:,1+N_zone*1:1+N_zone*2]/1000
    for ii in range(N_zone):
        P0[:,ii] = (data_in[:,3+N_zone*2+3*ii])/1000
        T_in0[:,ii] = data_in[:,1+N_zone*2+3*ii]
    T_o0 = data_in[:,0].reshape(-1,1)
    T_occ0 = data_in[:,1:1+N_zone*1].copy()/1000

    area = Building_excel.iloc[i_build,8:8+N_zone].tolist()
    TRUE_ratio = area/np.sum(area)
    P0n = np.sum(P0[:,:],1).reshape(-1,1)
    T_rad0n = np.sum(T_rad0[:,:],1).reshape(-1,1)
    T_occ0n = np.sum(T_occ0[:,:],1).reshape(-1,1)
    T_in = sum(TRUE_ratio[i]*T_in0[:,i] for i in range(N_zone)).reshape(-1,1)
    
    X_tr = np.hstack((T_o0[Train_s:Train_e,0:1],T_rad0n[Train_s:Train_e,:],T_occ0n[Train_s:Train_e,:]))
    P_tr = P0n[Train_s:Train_e,0:1]
    Y_tr = T_in[Train_s:Train_e,0:1]
    
    X_te = np.hstack((T_o0[Train_s2:Train_e2,0:1],T_rad0n[Train_s2:Train_e2,:],T_occ0n[Train_s2:Train_e2,:]))
    P_te = P0n[Train_s2:Train_e2,0:1]
    Y_te = T_in[Train_s2:Train_e2,0:1]
    return X_tr,P_tr,Y_tr,X_te,P_te,Y_te


def data_input_more(T_Fre,Train_s,Train_e,Train_s2,Train_e2,i_build,building_name,csv_):
    Building_excel = pd.ExcelFile('Data/#Introduction.xlsx').parse('Sheet1')
    Building_zone = Building_excel['zones'].tolist()
    N_zone = Building_zone[i_build]
    T_len = 8760*T_Fre
    AUX_sum = 5*N_zone+1
    if csv_ == 'yes':
        data_in = np.loadtxt('Data/Data_in/'+building_name+'.csv',delimiter=",",skiprows=1,usecols=range(1,AUX_sum+1))
    P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
    T_rad0 = data_in[:,1+N_zone*1:1+N_zone*2]/1000
    for ii in range(N_zone):
        P0[:,ii] = (data_in[:,3+N_zone*2+3*ii])/1000
        T_in0[:,ii] = data_in[:,1+N_zone*2+3*ii]
    T_o0 = data_in[:,0].reshape(-1,1)
    T_occ0 = data_in[:,1:1+N_zone*1].copy()/1000

    area = Building_excel.iloc[i_build,8:8+N_zone].tolist()
    TRUE_ratio = area/np.sum(area)
    P0n = np.sum(P0[:,:],1).reshape(-1,1)
    T_rad0n = np.sum(T_rad0[:,:],1).reshape(-1,1)
    T_occ0n = np.sum(T_occ0[:,:],1).reshape(-1,1)
    T_in = sum(TRUE_ratio[i]*T_in0[:,i] for i in range(N_zone)).reshape(-1,1)
    
    X_tr = np.hstack((T_o0[Train_s:Train_e,0:1],T_rad0[Train_s:Train_e,:],T_occ0[Train_s:Train_e,:]))
    P_tr = P0n[Train_s:Train_e,0:1]
    Y_tr = T_in[Train_s:Train_e,0:1]
    
    X_te = np.hstack((T_o0[Train_s2:Train_e2,0:1],T_rad0[Train_s2:Train_e2,:],T_occ0[Train_s2:Train_e2,:]))
    P_te = P0n[Train_s2:Train_e2,0:1]
    Y_te = T_in[Train_s2:Train_e2,0:1]
    return X_tr,P_tr,Y_tr,X_te,P_te,Y_te

def Inteplo(T_y, step):
    m,n = T_y.shape
    NN_step = (n-1)*step + 1
    T_y_n = torch.zeros((m,NN_step))
    for ii in range(n):
        T_y_n[:,step*ii] = T_y[:,ii]
    for j1 in range(n-1):
        for j2 in range(1,step):
            T_y_n[:,step*j1+j2] = T_y[:,j1] + j2*(T_y[:,j1+1]-T_y[:,j1])/step
    return T_y_n

def getdata(batch_step_bei, T_Fre, S_Y_tr,S_X_tr,S_P_tr, FF, n_zone):
    batch_size = int((S_P_tr.shape[0])/(24*T_Fre))
    batch_days = [i for i in range(batch_size)]
    batch_num = 24*T_Fre
    batch_time = torch.arange(0., batch_num, FF).to(device)

    T_Y_tr = np.zeros((batch_size,len(batch_time)))
    for i in range(batch_size):
        for j in range(len(batch_time)):
            T_Y_tr[i,j] = S_Y_tr[i*(batch_num)+FF*j,0]
    T_Y_tr = torch.tensor(T_Y_tr).to(device)

    input_num = S_X_tr.shape[1]  #1 + n_zone*2
    T_X_tr = np.zeros((input_num,batch_size,len(batch_time)))
    for i in range(input_num):
        for j in range(batch_size):
            for k in range(len(batch_time)):
                T_X_tr[i,j,k] = S_X_tr[j*(batch_num)+FF*k,i]
    T_X_tr = torch.tensor(T_X_tr).to(device)

    T_P_tr = np.zeros((1,batch_size,len(batch_time)))
    for i in range(batch_size):
        for j in range(len(batch_time)):
            T_P_tr[0,i,j] = S_P_tr[i*(batch_num)+FF*j,0]
    T_P_tr = torch.tensor(T_P_tr).to(device)

    T_y_tr = torch.cat((T_P_tr,T_X_tr),dim = 0)

    batch_y0 = torch.zeros((len(batch_days),1,1)).to(device)  #(21,1,1)
    batch_y0[:,0,0] = T_Y_tr[batch_days,0]
    NN_step = (len(batch_time)-1)*batch_step_bei + 1
    batch_y11 = torch.zeros((NN_step,len(batch_days),1,T_y_tr.shape[0])).to(device)   #(191,21,1,5)
    if batch_step_bei == 1:
        for i in range(len(batch_days)):
            batch_y11[:,i,0,:] = torch.transpose(T_y_tr[:,i,:],0,1)
    else:
        for i in range(len(batch_days)):
            batch_ytt = Inteplo(T_y_tr[:,batch_days[i],:], batch_step_bei)
            batch_y11[:,i,0,:] = torch.transpose(batch_ytt,0,1)
    batch_y = torch.zeros((len(batch_time),len(batch_days),1,1)).to(device)  #(96,21,1,1)
    for i in range(len(batch_days)):
        batch_y[:,i,0,0] = T_Y_tr[batch_days[i],:]
    return T_Y_tr, batch_y0, batch_y11, batch_y

def verify_test(Y_te_pre,Y_te):
    T_len = Y_te.shape[0]
    T_len1 = Y_te.shape[0]
    Err_tr = np.abs(Y_te_pre[:,0] - Y_te[:,0])
    Err_tr1 = math.sqrt(sum(Err_tr[i]**2/(T_len1) for i in range(T_len)))  #RMSE
    Err_tr2 = sum(Err_tr[i] for i in range(T_len))/(T_len1)  #MAE
    Err_tr3 = max(Err_tr)  #MAX
    Err_tr4 = r2_score(Y_te, Y_te_pre)  
    ERR2 = [Err_tr1,Err_tr2,Err_tr3,Err_tr4]
    return Err_tr,ERR2

def data_collect(i_build):
    Building_excel = pd.ExcelFile('Data/#Introduction.xlsx').parse('Sheet1')
    Building_name = Building_excel['name'].tolist()
    Building_zone = Building_excel['zones'].tolist()
    T_Fre = 4

    Train_s = T_Fre*24*(31+28+31+30+31)
    Train_e = Train_s + T_Fre*24*(30+31)
    Train_s2 = Train_e
    Train_e2 = Train_s2 + T_Fre*24*(31)

    X_tr1,P_tr1,Y_tr1,X_te1,P_te1,Y_te1 = data_input_more(T_Fre,Train_s,Train_e,Train_s2,Train_e2,i_build,Building_name[i_build],'yes')
    SS_X,SS_P,SS_Y = MinMaxScaler().fit(X_tr1),MinMaxScaler().fit(P_tr1),MinMaxScaler().fit(Y_tr1)
    S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te = SS_X.transform(X_tr1),SS_P.transform(P_tr1),SS_Y.transform(Y_tr1),SS_X.transform(X_te1),SS_P.transform(P_te1),SS_Y.transform(Y_te1)
    
    FF_tr = 1
    FF_te = 1
    batch_step_bei_tr = 1
    batch_step_bei_te = 1
    T_Y_tr, batch_y0, batch_y11, batch_y = getdata(batch_step_bei_tr, T_Fre,S_Y_tr,S_X_tr,S_P_tr, FF_tr, Building_zone[i_build])
    T_Y_te, batch_t_y0, batch_t_y11, batch_t_y = getdata(batch_step_bei_te, T_Fre,S_Y_te,S_X_te,S_P_te, FF_te, Building_zone[i_build])

    ND_u_tr,ND_u_te = batch_y11.permute(3,1,0,2).squeeze(),batch_t_y11.permute(3,1,0,2).squeeze()
    Time = ND_u_tr.shape[-1]
    ND_y = torch.linspace(0, 1, Time).view(Time, 1).to(device)
    T_Y_tr_,T_Y_te_ = copy.deepcopy(T_Y_tr),copy.deepcopy(T_Y_te)
    for i in range(T_Y_tr.shape[0]):
        T_Y_tr_[i,:] = T_Y_tr[i,:] + (0-T_Y_tr[i,0])
    for i in range(T_Y_te.shape[0]):
        T_Y_te_[i,:] = T_Y_te[i,:] + (0-T_Y_te[i,0])
    
    Data = {}
    Data[str(i_build)] = [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y]

    build_DR = ['1','2','3','4','5','6']
    build_name = ['1_new','2_new','3_new','1','2','3']
    
    for iii in range(len(build_DR)):
        build_DR_name = Building_name[i_build]+'_'+build_name[iii]
        _,_,_,C_X_te1,C_P_te1,C_Y_te1 = data_input_more(T_Fre,Train_s,Train_e,Train_s2,Train_e2,i_build,build_DR_name,'yes')
        C_S_X_te,C_S_P_te,C_S_Y_te = SS_X.transform(C_X_te1),SS_P.transform(C_P_te1),SS_Y.transform(C_Y_te1)
        C_T_Y_te, C_batch_t_y0, C_batch_t_y11, C_batch_t_y = getdata(batch_step_bei_te, T_Fre,C_S_Y_te,C_S_X_te,C_S_P_te, FF_te, Building_zone[i_build])
        C_ND_u_te = C_batch_t_y11.permute(3,1,0,2).squeeze()
        C_T_Y_te_ = copy.deepcopy(C_T_Y_te)
        for i in range(C_T_Y_te.shape[0]):
            C_T_Y_te_[i,:] = C_T_Y_te[i,:] + (0-C_T_Y_te[i,0])
        Data[str(i_build)+'_'+build_DR[iii]] = [C_Y_te1,C_T_Y_te,C_T_Y_te_,C_ND_u_te,   C_batch_t_y0,C_batch_t_y11,C_batch_t_y]    
    return Data,SS_Y
