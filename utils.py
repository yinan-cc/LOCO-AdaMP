import numpy as np
from ml_models import *

def get_mn(M,N):
    n = int(N**0.8)
    #m = int(min(max(0.1*M, 15),0.5*n))
    m = int(4*np.log(M))
    return m,n

def get_Kb(M,N,m,n,k = [100, 100, 100, 100, 1000]):
    c = M/m * N/n
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    k4 = k[3]
    k5 = k[4]
    if M == 50:
        k1 = k2 = k3 = k4 = 100
        k5 = 1000
    if M == 500:
        k1 = k2 = k3 = k4 = 50
        k5 = 250
    #Kb = [int(k1* c), int(k2* c), int(k3* c), int(k4* c), int(k5* c)]
    Kb = [2000, 2000, 2000, 2000, 10000]
    return Kb

def get_parameter(M, N):
    m,n = get_mn(M,N)
    print(f"M = {M}, N = {N}, m = {m}, n = {n}")

    Kb = get_Kb(M,N,m,n)
    K_locomp = sum(Kb)
    print(f"Kb = {Kb}, Kb_sum = {K_locomp}")

def get_betas(beta1, beta2):
    betas = []
    for i in beta1:
        betas.append(beta2+[i])
    return betas

def get_model(name):
    if name == "Ridge":
        model = RidgeReg
    if name == "RandomForest":
        model  = DecisionTreeReg
    if name == "KernelRidge":
        model = KernelRidgeReg
    return model

def get_model_split(name):
    if name == "Ridge":
        model = RidgeCVReg
    if name == "RandomForest":
        model  = RandomForestReg
    if name == "KernelRidge":
        model = KernelRidgeCVReg
    return model

def get_model_name(model):
    if (model == RidgeReg) or (model == RidgeCVReg):
        model_name ="Ridge"
    if (model == DecisionTreeReg) or (model == RandomForestReg):
        model_name  = "RandomForest"
    if (model == KernelRidgeReg) or (model == KernelRidgeCVReg):
        model_name = "KernelRidge"
    return model_name

def format_simu_type(type_in):
    if type_in == "independent":
        type_out = "Independent"
    if type_in == "correlated":
        type_out = "Correlated"   
    if type_in == "nonlinear":
        type_out = "Nonlinear"
    return type_out

def format_ml_name(name_in):
    if name_in == "ridge":
        name_out = "Ridge"
    if name_in == "randomforest":
        name_out = "RandomForest"   
    if name_in == "kernelridge":
        name_out = "KernelRidge"
    return name_out

def get_M(dim):
    if dim == "low":
        M = 50
    if dim == "high":
        M = 500
    return M

def get_seed(simu_type, ml_name=None, lossfunc=None):
    if ml_name is None:
        m = 0
    if ml_name == "Ridge":
        m = 1
    if ml_name == "RandomForest":
        m = 2
    if ml_name == "KernelRidge":
        m = 3
    if lossfunc is None:
        l = 4
    if lossfunc == "sq":
        l = 5
    if lossfunc == "abs":
        l = 6
    if simu_type == "Independent":
        s = 7
    if simu_type == "Correlated":
        s = 8
    if simu_type == "Nonlinear":
        s = 9
    return s*123+m*37+l*99

def get_seed0(seed0_in):
    if seed0_in is None:
        seed0 = np.random.randint(1,100000)
    else:
        seed0 = seed0_in
    return seed0