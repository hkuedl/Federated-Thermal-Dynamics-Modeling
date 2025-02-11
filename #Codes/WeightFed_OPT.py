#%%
import numpy as np
import math
import pandas as pd
import os
import random
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
from torch.autograd import grad
device = torch.device("cuda:2")
import Common_Func
import WeightFed_Func
import cvxpy as cp

def to_np(x):
    return x.cpu().detach().numpy()

T_Fre = 4

Building_excel = pd.ExcelFile('Data/#Introduction.xlsx').parse('Sheet1')
Building_name = Building_excel['name'].tolist()
Building_high = Building_excel['point1'].tolist()
Building_low = Building_excel['point2'].tolist()
Building_tem_bas = [0.5*(Building_high[i]+Building_low[i]) for i in range(len(Building_high))]

Data_all = []
for i_build in range(len(Building_name)):
    Data,_ = Common_Func.data_collect(i_build)
    Data_all.append(Data)

Build_num = len(Data_all)

Train_s = T_Fre*24*(31+28+31+30+31)
Train_e = Train_s + T_Fre*24*(30+31)
Train_s2 = Train_e
Train_e2 = Train_s2 + T_Fre*24*(31)

Methods = ['Single','fedavg','model','data','fix_w','both']

for i_build in range(Build_num):
    X_tr1,P_tr1,Y_tr1,X_te1,P_te1,Y_te1 = Common_Func.data_input_more(T_Fre,Train_s,Train_e,Train_s2,Train_e2,i_build,Building_name[i_build],'yes')
    SS_X,SS_P,SS_Y = MinMaxScaler().fit(X_tr1),MinMaxScaler().fit(P_tr1),MinMaxScaler().fit(Y_tr1)

    [Y_tr1,T_Y_tr,T_Y_tr_,ND_u_tr,Y_te1,T_Y_te,T_Y_te_,ND_u_te,ND_y,  batch_y0,batch_y11,batch_y,batch_t_y0,batch_t_y11,batch_t_y] = Data_all[i_build][str(i_build)]

    input_num = Data_all[i_build][str(i_build)][10].shape[-1]-1
    output_num_mid = 20
    hidden_units = 20
    p_input_num = 20
    p_output_num = 1
    p_hidden_units = 20
    local_model = WeightFed_Func.NeuralODE_per_newlayer(output_num_mid, output_num_mid, hidden_units,2).to(device)
    phy_model = WeightFed_Func.Federated_phy_newlayer(p_input_num, p_output_num, p_hidden_units,2).to(device)
    merge_model = WeightFed_Func.NeuralODE_merge(local_model,phy_model).to(device)

    merge_model_final = WeightFed_Func.NeuralODE_merge_final(input_num,output_num_mid,merge_model).to(device)
    #Here please choose any comparison method to load the NN models
    
    #m_state_dict = torch.load('#Results/Models_single/'+'Neuralode_'+str(i_build)+'.pt')
    #m_state_dict = torch.load('#Results/Models_fedavg/'+'Neuralode_'+str(i_build)+'.pt')
    #m_state_dict = torch.load('#Results/Models_oneW/'+'Neuralode_'+str(i_build)+'_model.pt')
    #m_state_dict = torch.load('#Results/Models_oneW/'+'Neuralode_'+str(i_build)+'_data.pt')
    #m_state_dict = torch.load('#Results/Models_oneW/'+'Neuralode_'+str(i_build)+'_fix.pt')
    m_state_dict = torch.load('#Results/Models_newcomb/'+'Neuralode_'+str(i_build)+'.pt')
    
    merge_model_final.load_state_dict(m_state_dict)

    batch_time_tr = torch.arange(0., 24*4, 1).to(device)
    batch_step_tr = batch_time_tr[1] - batch_time_tr[0]

    fre_opt = 1
    fre_cal = 1
    c_dT = int(fre_cal*fre_opt)
    c_0_price = np.array(pd.read_excel('Data/Price_signal.xlsx'),dtype=float)
    c_0_tem_comfort = [1.6,1.6,1.6,1.6,1.6,1.6,1.2,1.2,1.2,1.2,0.8,0.8,0.8,1.0,1.0,1.0,1.0,1.0,1.6,1.6,1.6,1.6,1.6,1.6]
    c_0_tem_comfort = [i*0.5 for i in c_0_tem_comfort]

    c_time_p = int(96/fre_opt)
    c_time = int(96/fre_opt) + 1
    c_p_A,c_p_B,c_p_F = cp.Parameter(),cp.Parameter(),cp.Parameter(c_time-1)
    c_v_tem,c_v_q = cp.Variable(c_time),cp.Variable(c_time_p)
    c_v_tem_u,c_v_tem_l = cp.Variable(c_time, pos=True),cp.Variable(c_time, pos=True)

    c_upper_tem = np.zeros((c_time,1))
    c_lower_tem = np.zeros((c_time,1))
    c_upper_p = 40*np.ones((c_time_p,1))
    c_lower_p = 0*np.ones((c_time_p,1))
    c_price = np.zeros((c_time_p,1))
    for i in range(24):
        for j in range(int(c_time_p/24)):
            c_price[i*int(c_time_p/24)+j,0] = 0.001*c_0_price[i,1]
            c_lower_tem[i*int(c_time_p/24)+j,0] = Building_tem_bas[i_build]-c_0_tem_comfort[i]
            c_upper_tem[i*int(c_time_p/24)+j,0] = Building_tem_bas[i_build]+c_0_tem_comfort[i]
    c_lower_tem[-1,0],c_upper_tem[-1,0] = Building_tem_bas[i_build]-c_0_tem_comfort[0], Building_tem_bas[i_build]+c_0_tem_comfort[0]

    c_upper_tem = SS_Y.transform(c_upper_tem).ravel()
    c_lower_tem = SS_Y.transform(c_lower_tem).ravel()
    c_PI = 3.6
    c_upper_q = SS_P.transform(c_PI*c_upper_p).ravel()
    c_lower_q = SS_P.transform(c_PI*c_lower_p).ravel()

    c_initial_tem = c_lower_tem[0]

    c_Pmax,c_Pmin,c_tmax,c_tmin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]
    c_tem_cost_u,c_tem_cost_l = 1.0*(c_tmax-c_tmin), 1.0*(c_tmax-c_tmin)
    
    c_obj1 = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p))
    c_obj2 = sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))

    c_obj = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p)) \
        +sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))

    cons_tem1 = [c_v_tem[fre_cal*t] <= c_upper_tem[t]+c_v_tem_u[t] for t in range(c_time)]
    cons_tem2 = [c_v_tem[fre_cal*t] >= c_lower_tem[t]-c_v_tem_l[t] for t in range(c_time)]
    cons_q = [c_v_q <= c_upper_q, c_v_q >= c_lower_q]
    cons_model = [c_v_tem[t+1] == c_v_tem[t] + (1/fre_cal)*(c_p_A*c_v_tem[t] + c_p_B*c_v_q[int(t//c_dT)] + c_p_F[t]) for t in range(c_time-1)]
    cons_ini = [c_v_tem[0] == c_lower_tem[0]]

    cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini

    c_prob = cp.Problem(cp.Minimize(c_obj), cons)

    
    Cr_obj,Cr_q,Cr_tem,Cr_temu,Cr_teml = [],[],[],[],[]
    Cr_obj1,Cr_obj2 = [],[]
    Ce_obj1,Ce_obj2 = [],[]
    Ce_obj,Ce_q,Ce_tem,Ce_temu,Ce_teml = [],[],[],[],[]
    Cr_q_hy,Cr_tem_hy,Ce_q_hy,Ce_tem_hy = [],[],[],[]
    train_days = batch_y0.shape[0]
    test_days = batch_t_y0.shape[0]
    c_p_A.value = merge_model_final.nn_merge.a.item()
    c_p_B.value = merge_model_final.nn_merge.b.item()

    comp2_pre = merge_model_final.prepross(batch_y11[:,:,:,1:])
    comp2 = merge_model_final.nn_merge(comp2_pre)

    comp2_pre_t = merge_model_final.prepross(batch_t_y11[:,:,:,1:])
    comp2_t = merge_model_final.nn_merge(comp2_pre_t)

    for d in range(train_days):
        c_p_F.value = to_np(comp2[:,d,0,0])
        c_prob.solve(solver=cp.GUROBI,verbose=False)
        Cr_obj.append(c_prob.value)
        Cr_obj1.append(c_obj1.value)
        Cr_obj2.append(c_obj2.value)
        Cr_q.append(c_v_q.value)
        Cr_q_hy.append(SS_P.inverse_transform(c_v_q.value.reshape(-1,1)).ravel())
        Cr_tem.append(c_v_tem.value)
        Cr_tem_hy.append(SS_Y.inverse_transform(c_v_tem.value.reshape(-1,1)).ravel())
        Cr_temu.append(c_v_tem_u.value)
        Cr_teml.append(c_v_tem_l.value)

    for d in range(test_days):
        c_p_F.value = to_np(comp2_t[:,d,0,0])
        c_prob.solve(solver=cp.GUROBI,verbose=False)
        Ce_obj.append(c_prob.value)
        Ce_obj1.append(c_obj1.value)
        Ce_obj2.append(c_obj2.value)
        Ce_q.append(c_v_q.value)
        Ce_q_hy.append(SS_P.inverse_transform(c_v_q.value.reshape(-1,1)).ravel())
        Ce_tem.append(c_v_tem.value)
        Ce_tem_hy.append(SS_Y.inverse_transform(c_v_tem.value.reshape(-1,1)).ravel())
        Ce_temu.append(c_v_tem_u.value)
        Ce_teml.append(c_v_tem_l.value)
        #print("status:", c_prob.status)

    Cr_objsum = sum(Cr_obj)
    Ce_objsum = sum(Ce_obj)
    Cr_objsum1 = sum(Cr_obj1)
    Ce_objsum1 = sum(Ce_obj1)
    Cr_objsum2 = sum(Cr_obj2)
    Ce_objsum2 = sum(Ce_obj2)
    print('Train objective:')
    print(Cr_objsum)
    print('Test objective:')
    print(Ce_objsum)

    import pandas as pd
    #writer = pd.ExcelWriter('#Results/'+'Local_'+str(i_build)+'_up.xlsx')
    #writer = pd.ExcelWriter('#Results/'+'SW_avg_'+str(i_build)+'_up.xlsx')
    #writer = pd.ExcelWriter('#Results/'+'SW_model_'+str(i_build)+'_up.xlsx')
    #writer = pd.ExcelWriter('#Results/'+'SW_data_'+str(i_build)+'_up.xlsx')
    #writer = pd.ExcelWriter('#Results/'+'DW_fix_'+str(i_build)+'_up.xlsx')
    writer = pd.ExcelWriter('#Results/'+'DW_ada_'+str(i_build)+'_up.xlsx')
    
    to_Cr_q = pd.DataFrame(Cr_q_hy)
    to_Cr_q.to_excel(writer,sheet_name='q_tr',index=False)
    to_Cr_tem = pd.DataFrame(Cr_tem_hy)
    to_Cr_tem.to_excel(writer,sheet_name='tem_tr',index=False)
    to_Ce_q = pd.DataFrame(Ce_q_hy)
    to_Ce_q.to_excel(writer,sheet_name='q_te',index=False)
    to_Ce_tem = pd.DataFrame(Ce_tem_hy)
    to_Ce_tem.to_excel(writer,sheet_name='tem_te',index=False)

    writer.close()
