#%%
import numpy as np
import math
import pandas as pd
import os
import random
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import copy
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

class Weight_Server():
    def __init__(self,clients,fed_epoch,local_e1,local_e2,similar_matrix,Med_type):
        self.clients = clients
        self.fed_epoch = fed_epoch
        self.local_e1,self.local_e2 = local_e1,local_e2
        self.similar_matrix = similar_matrix
        self.Med_type = Med_type

    def fed_train(self,weight):
        print('Federated Training!')
        for i in range(len(self.clients)):
            self.clients[i].set_model(self.clients[0].get_model())
            
        loss_erm = np.zeros((len(self.clients),self.fed_epoch+1))
        
        for e in range(self.fed_epoch):
            print(f'Federated training [Epoch {e}/{self.fed_epoch}]')
            # Client training
            for i_client in range(len(self.clients)):
                loss_erm[i_client,e] = self.clients[i_client].fed_local_train(self.local_e1)
            # Model weighting
            global_params = self.model_weighting()
            # Model distribute
            for i in range(len(self.clients)):
                self.clients[i].set_model(global_params[i])
        for i_client in range(len(self.clients)):
            if self.Med_type == 'fix':
                torch.save(self.clients[i_client].merge_model_final.state_dict(), '#Results/Models_oneW/'+NN_type+'_'+str(i_client)+'_fix.pt')
            elif self.Med_type == 'data':
                torch.save(self.clients[i_client].merge_model_final.state_dict(), '#Results/Models_oneW/'+NN_type+'_'+str(i_client)+'_data.pt')
            elif self.Med_type == 'model': 
                torch.save(self.clients[i_client].merge_model_final.state_dict(), '#Results/Models_oneW/'+NN_type+'_'+str(i_client)+'_model.pt')
        return loss_erm

    def model_weighting(self):
        local_models = []
        all_model_params = []
        for client in self.clients:
            local_models.append(client.get_model())
        for i_model in range(len(local_models)):
            all_model_params.append(local_models[0])
            for param_name in all_model_params[i_model]:
                all_model_params[i_model][param_name] = self.similar_matrix[i_model,0]*local_models[0][param_name]
                for ii in range(1,len(local_models)):
                    all_model_params[i_model][param_name] += self.similar_matrix[i_model,ii]*local_models[ii][param_name]
        return all_model_params

class Weight_Client():
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
    
    def get_model(self):
        return self.merge_model.state_dict()
    
    def set_model(self, model_params):
        self.merge_model.load_state_dict(model_params)

    def fed_local_train(self,new_epoch):
        optimizer = optim.Adam(self.merge_model_final.parameters(), lr=self.lr)
        epoch_freq = int(self.fed_local_epoch/5)
        [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y] = self.Data[str(self.id)]
        batch_size = ND_u_tr.shape[-2]
        batch_num = math.ceil(ND_u_tr.shape[-2]/batch_size)

        n_batch_list_all = []
        for epoch in range(new_epoch):
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
        return loss.cpu().detach().numpy()
    def accuracy_eva(self,SS_Y):
        [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y] = self.Data[str(self.id)]
        SS_Y = MinMaxScaler().fit(Y_tr1)
        def ACC(SS_Y,merge_model,Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,ND_y,batch_y0, batch_y11):
            if self.NN_type == 'Neuralode':
                batch_time_tr = torch.arange(0., 24*4, 1).to(device)
                batch_step_tr = batch_time_tr[1] - batch_time_tr[0]
                y_input, y_phy = 1,0
                pred_y = odeint(merge_model, batch_y0, batch_y11, y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
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

Building_feature = pd.ExcelFile('Data/#Introduction.xlsx').parse('Feature')
Building_feature = np.array(Building_feature.iloc[:,1:])
SS_X = MinMaxScaler().fit(Building_feature.T)
Building_feature_01 = SS_X.transform(Building_feature.T)
Building_dist = distance.cdist(Building_feature_01, Building_feature_01, metric='euclidean')
Building_simi = 1/(1+Building_dist)
for i in range(Building_simi.shape[0]):
    Building_simi[i,i] = 0
Building_simi_1 = np.zeros((Building_simi.shape[0],Building_simi.shape[1]))
for i in range(Building_simi_1.shape[0]):
    for j in range(Building_simi_1.shape[1]):
        if i == j:
            Building_simi_1[i,j] = 1/32
        else:
            Building_simi_1[i,j] = (31/32)*Building_simi[i,j]/sum(Building_simi[i,:])

Building_simi_data = np.array(pd.read_excel('Data/Domain_similarity_temper.xlsx',header=None))
Building_simi_2_0 = 1/(1+Building_simi_data)
for i in range(Building_simi_2_0.shape[0]):
    Building_simi_2_0[i,i] = 0
Building_simi_2_1 = np.zeros((Building_simi_2_0.shape[0],Building_simi_2_0.shape[1]))
for i in range(Building_simi_2_1.shape[0]):
    for j in range(Building_simi_2_1.shape[1]):
        if i == j:
            Building_simi_2_1[i,j] = 1/32
        else:
            Building_simi_2_1[i,j] = (31/32)*Building_simi_2_0[i,j]/sum(Building_simi_2_0[i,:])

weight = 0.5
Building_simi_final = weight*Building_simi_2_1 + (1-weight)*Building_simi_1

Build_num = len(Data_all)
N_data_type = 8
ACC_result_RMSE = np.zeros((Build_num,N_data_type))
ACC_result_MAE = np.zeros((Build_num,N_data_type))
ACC_result_R2 = np.zeros((Build_num,N_data_type))
ACC_result_All0 = np.zeros((3,Build_num,N_data_type))
ACC_result_curve = np.zeros((Build_num,Data_all[0]['0_1'][0].shape[0],2*6+2))

NN_type_list = ['Neuralode','Deeponet','FNN']
Med_type = 'fix'  #'model','data','fix'
###Here please choose any Med_type to run comparison methods: SW_model, SW_data, DW_fix
for ii in [0]:
    NN_type = NN_type_list[ii]
    if NN_type == NN_type_list[0]:
        fed_epoch = 50
        local_e1,local_e2 = 20,5
        fed_local_epoch = 40
        NN_num = 4
        hidden_units = 20
        hidden_units_trunk = 20
        p_hidden_units = 20
        output_num_mid = p_input_num = 20
        p_output_num = output_num_trunk = 1
        lr = 1e-2

    clients =[]
    for i_build in range(len(Data_all)):
        input_num = Data_all[i_build][str(i_build)][10].shape[-1]-1
        client = Weight_Client(i_build,Data_all[i_build],input_num,output_num_mid,output_num_trunk,hidden_units,hidden_units_trunk,NN_num,lr,fed_local_epoch,p_input_num,p_output_num,p_hidden_units,NN_type)
        clients.append(client)
    if Med_type == 'fix':
        server = Weight_Server(clients,fed_epoch,local_e1,local_e2,Building_simi_final)
    elif Med_type == 'data':
        server = Weight_Server(clients,fed_epoch,local_e1,local_e2,Building_simi_2_1)
    elif Med_type == 'model':
        server = Weight_Server(clients,fed_epoch,local_e1,local_e2,Building_simi_1)
    
    loss_erm = server.fed_train(weight)
    for i_build in range(len(clients)):
        ACC_result_i,ACC_result_curve[i_build,:,:] = clients[i_build].accuracy_eva(SS_Y)
        ACC_result_RMSE[i_build,:] = ACC_result_i[:,0]
        ACC_result_MAE[i_build,:]  = ACC_result_i[:,1]
        ACC_result_R2[i_build,:]   = ACC_result_i[:,3]
        ACC_result_All0[0,i_build,:] = ACC_result_RMSE[i_build,:]
        ACC_result_All0[1,i_build,:] = ACC_result_MAE[i_build,:]
        ACC_result_All0[2,i_build,:] = ACC_result_R2[i_build,:]

loss_erm_mean = np.mean(loss_erm,axis = 0)

#%% save

writer = pd.ExcelWriter('#Results/Accu_'+Med_type+'_new.xlsx', engine='xlsxwriter')
sheet_name = ['RMSE','MAE','R2']
for i in range(ACC_result_All0.shape[0]):
    df = pd.DataFrame(ACC_result_All0[i,:,:])
    df.to_excel(writer, sheet_name=sheet_name[i], index=False, header=False)
writer.close()

writer1 = pd.ExcelWriter('#Results/Accu_'+Med_type+'_new_curve.xlsx', engine='xlsxwriter')
sheet_name1 = [str(i) for i in range(len(Data_all))]
for i in range(ACC_result_curve.shape[0]):
    df = pd.DataFrame(ACC_result_curve[i,:,:])
    df.to_excel(writer1, sheet_name=sheet_name1[i], index=False, header=False)
writer1.close()
