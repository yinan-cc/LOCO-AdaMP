from ml_models import *

def get_mn(M,N):
    n = int(N**0.8)
    m = int(min(max(0.1*M, 15),0.5*n))
    return m,n

def get_Kb(M,N,m,n):
    c = M/m * N/n
    if M == 50:
        k1 = k2 = k3 = k4 = 100
        k5 = 1000
    if M == 500:
        k1 = k2 = k3 = k4 = 50
        k5 = 250
    Kb = [int(k1* c), int(k2* c), int(k3* c), int(k4* c), int(k5* c)]
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