import numpy as np
from scipy.stats import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def RidgeReg():
    model = Ridge(fit_intercept = False, alpha=0.001)
    return model

def RidgeCVReg():
    alphas_logspace = np.logspace(-3, 1, 50)
    model = RidgeCV(alphas=alphas_logspace, fit_intercept = False, cv = 5)
    return model

def DecisionTreeReg():
    seed = np.random.randint(1,100000)
    model = DecisionTreeRegressor(random_state=seed)
    return model

def RandomForestReg():
    seed = np.random.randint(1,100000)
    model= RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
    return model

def KernelRidgeReg():
    model = KernelRidge(alpha=0.001, kernel='rbf')
    return model

def KernelRidgeCVReg():
    model0 = KernelRidge(kernel='rbf')
    alphas_logspace = np.logspace(-3, 1, 50)
    param_grid = {'alpha': alphas_logspace}
    model = GridSearchCV(estimator=model0, param_grid=param_grid, cv=5)
    return model