#%%
import numpy as np
import math
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
import torch

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

set_seed(20)

import torch.nn as nn
import torch.optim as optim
from torchdiffeq1 import odeint
from torch.autograd import grad
device = torch.device("cuda:2")
import matplotlib.pyplot as plt
import Common_Func
import WeightFed_Func

class Single_Client():
    def __init__(self,id,Data,input_num,output_num_mid,output_num_trunk,hidden_units,hidden_units_trunk,NN_num,lr,fed_local_epoch,p_input_num,p_output_num,p_hidden_units,NN_type):
        self.id = id
        self.input_num = input_num
        self.output_num_mid = output_num_mid
        self.hidden_units = hidden_units
        self.output_num_trunk = output_num_trunk
        self.hidden_units_trunk = hidden_units_trunk
        self.NN_num = NN_num
        self.lr = lr
        self.Data = Data

        self.p_input_num = p_input_num
        self.p_output_num = p_output_num
        self.p_hidden_units = p_hidden_units
        
        self.fed_local_epoch = fed_local_epoch
        
        self.NN_type = NN_type

        if self.NN_type == 'Neuralode':
            self.local_model = WeightFed_Func.NeuralODE_per_newlayer(self.output_num_mid, self.output_num_mid, self.hidden_units,2).to(device)
        
        self.phy_model = WeightFed_Func.Federated_phy_newlayer(self.p_input_num, self.p_output_num, self.p_hidden_units,2).to(device)

        if self.NN_type == 'Neuralode':
            self.merge_model = WeightFed_Func.NeuralODE_merge(self.local_model,self.phy_model).to(device)
            self.merge_model_final = WeightFed_Func.NeuralODE_merge_final(self.input_num,self.output_num_mid,self.merge_model).to(device)

    def fed_local_train(self,index):
        #set_seed(20)
        if index == 'free':
            optimizer = optim.Adam(self.merge_model_final.parameters(), lr=self.lr)
        epoch_freq = int(self.fed_local_epoch/5)
        [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y] = self.Data[str(self.id)]
        batch_size = ND_u_tr.shape[-2]
        batch_num = math.ceil(ND_u_tr.shape[-2]/batch_size)

        n_batch_list_all = []
        for epoch in range(self.fed_local_epoch):
            batch_list = list(range(ND_u_tr.shape[-2]))
            for num in range(batch_num):
                n_batch_list = random.sample(batch_list,min(batch_size,len(batch_list)))
                batch_list = [x for x in batch_list if x not in n_batch_list]
                n_batch_list_all.append(n_batch_list)
                optimizer.zero_grad()
                if self.NN_type == 'Neuralode':
                    batch_time_tr = torch.arange(0., 24*4, 1).to(device)
                    batch_step_tr = batch_time_tr[1] - batch_time_tr[0]
                    n_batch_y0 = batch_y0[n_batch_list,:,:]
                    n_batch_y11 = batch_y11[:,n_batch_list,:,:]
                    n_batch_y = batch_y[:,n_batch_list,:,:]
                    y_input, y_phy = 1,0
                    n_pred_y = odeint(self.merge_model_final, n_batch_y0, n_batch_y11,y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
                    n_pred_y.to(device)
                    loss = torch.mean((n_pred_y - n_batch_y)**2)
                loss.backward()
                optimizer.step()
            if epoch % epoch_freq == 0:
               with torch.no_grad():
                   print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch,(num+1)*batch_size,ND_u_tr.shape[-2],loss.item()))
    
    def accuracy_eva(self,SS_Y):
        [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y] = self.Data[str(self.id)]
        SS_Y = MinMaxScaler().fit(Y_tr1)
        def ACC(SS_Y,merge_model,Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,ND_y,batch_y0, batch_y11):
            if self.NN_type == 'Neuralode':
                batch_time_tr = torch.arange(0., 24*4, 1).to(device)
                batch_step_tr = batch_time_tr[1] - batch_time_tr[0]
                y_input, y_phy = 1,0
                pred_y = odeint(merge_model, batch_y0, batch_y11,y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
                pred_y = pred_y.cpu().detach().numpy()
                pred_y_ = np.squeeze(pred_y)
                train_pred = pred_y_.T.flatten().reshape(-1,1)
                train_pred_hy = SS_Y.inverse_transform(train_pred)
                _,train_Err2 = Common_Func.verify_test(train_pred_hy,Y_tr1)
            return train_Err2,[Y_tr1,train_pred_hy]
        N_data_type = 8
        ACC_result = np.zeros((N_data_type,4))
        build_DR = ['1','2','3','4','5','6']
        build_curve = np.zeros((self.Data[str(self.id)+'_1'][0].shape[0],2*(len(build_DR)+1)))
        aaa,_ = ACC(SS_Y,self.merge_model_final,Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,ND_y,batch_y0, batch_y11)
        ACC_result[0,:] = np.array(aaa)
        bbb,build_curve_i = ACC(SS_Y,self.merge_model_final,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,batch_t_y0,batch_t_y11)
        ACC_result[1,:] = np.array(bbb)
        build_curve[:,0:1],build_curve[:,1:2] = build_curve_i[0],build_curve_i[1]
        for i in range(len(build_DR)):
            [C_Y_te1,C_T_Y_te,C_T_Y_te_,C_ND_u_te,C_batch_t_y0,C_batch_t_y11,C_batch_t_y] = self.Data[str(self.id)+'_'+build_DR[i]]
            ccc,build_curve_i = ACC(SS_Y,self.merge_model_final,C_Y_te1,C_T_Y_te,C_T_Y_te_,C_ND_u_te,ND_y,C_batch_t_y0,C_batch_t_y11)
            ACC_result[2+i,:] = np.array(ccc)
            build_curve[:,2+2*i:2+2*i+1],build_curve[:,2+2*i+1:2+2*i+2] = build_curve_i[0],build_curve_i[1]
        return ACC_result,build_curve

Building_excel = pd.ExcelFile('Data/#Introduction.xlsx').parse('Sheet1')
Building_name = Building_excel['name'].tolist()

Data_all = []
for i_build in range(len(Building_name)):
    Data,SS_Y = Common_Func.data_collect(i_build)
    Data_all.append(Data)

Build_num = len(Data_all)
N_data_type = 8
ACC_result_RMSE = np.zeros((Build_num,N_data_type))
ACC_result_MAE = np.zeros((Build_num,N_data_type))
ACC_result_R2 = np.zeros((Build_num,N_data_type))
ACC_result_All = np.zeros((3,Build_num,N_data_type))
ACC_result_curve = np.zeros((Build_num,Data_all[0]['0_1'][0].shape[0],2*6+2))

import time
NN_type_list = ['Neuralode','Deeponet','FNN']
for ii in [0]:
    NN_type = NN_type_list[ii]
    if NN_type == NN_type_list[0]:
        NN_num = 4
        hidden_units = 20
        hidden_units_trunk = 20
        p_hidden_units = 20
        output_num_mid = p_input_num = 20
        p_output_num = output_num_trunk = 1
        lr = 1e-2

    for i_build in range(0,len(Data_all)):
        if i_build == 2:
            fed_local_epoch = 900
        else:
            fed_local_epoch = 1000

        time1 = time.time()
        input_num = Data_all[i_build][str(i_build)][10].shape[-1]-1
        client = Single_Client(i_build,Data_all[i_build],input_num,output_num_mid,output_num_trunk,hidden_units,hidden_units_trunk,NN_num,lr,fed_local_epoch,p_input_num,p_output_num,p_hidden_units,NN_type)
        client.fed_local_train('free')
        time2 = time.time()
        print(time2-time1)
        ACC_result_i,ACC_result_curve[i_build,:,:] = client.accuracy_eva(SS_Y)
        ACC_result_RMSE[i_build,:] = ACC_result_i[:,0]
        ACC_result_MAE[i_build,:]  = ACC_result_i[:,1]
        ACC_result_R2[i_build,:]   = ACC_result_i[:,3]
        ACC_result_All[0,i_build,:] = ACC_result_RMSE[i_build,:]
        ACC_result_All[1,i_build,:] = ACC_result_MAE[i_build,:]
        ACC_result_All[2,i_build,:] = ACC_result_R2[i_build,:]

        print(ACC_result_RMSE[i_build,2])

print(np.mean(ACC_result_RMSE[:,:],axis=0))
