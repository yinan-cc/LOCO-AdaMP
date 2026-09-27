import numpy as np
from ml_models import *

def get_parameter(M, N):
    n = int(N**0.8)
    m = int(4*np.log(M))
    Kb = [2000, 2000, 2000, 2000, 10000]
    K_locomp = sum(Kb)
    return m,n,Kb,K_locomp

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

def ml_rename(ml_name):
    if ml_name == "RandomForest":
        ml_name = "DecisionTree/RF"
    return ml_name

def ml_rename1(ml_name):
    if ml_name == "RandomForest":
            ml_name = "DecisionTree"
    return ml_name


def simu_type_rename(simu_type):
    if simu_type == "Independent":
        simu_type = "Linear"
    elif simu_type == "Correlated":
        simu_type = "Linear Correlated"
    return simu_type

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