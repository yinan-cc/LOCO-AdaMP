import numpy as np
from scipy.stats import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def RidgeReg(X,Y,X1):
    model = Ridge(fit_intercept = False, alpha=0.001).fit(X, Y)
    return model.predict(X1)

def RidgeCVReg(X,Y,X1):
    alphas_logspace = np.logspace(-3, 1, 50)
    model = RidgeCV(alphas=alphas_logspace, fit_intercept = False, cv = 5).fit(X, Y)
    return model.predict(X1)

def DecisionTreeReg(X,Y,X1):
    model = DecisionTreeRegressor().fit(X,Y)
    return model.predict(X1)

def RandomForestReg(X,Y,X1):
    model= RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, bootstrap=False).fit(X,Y)
    return model.predict(X1)

def KernelRidgeReg(X,Y,X1):
    model = KernelRidge(alpha=0.001, kernel='rbf').fit(X, Y)
    return model.predict(X1)

def KernelRidgeCVReg(X,Y,X1):
    model = KernelRidge(kernel='rbf')
    alphas_logspace = np.logspace(-3, 1, 50)
    param_grid = {'alpha': alphas_logspace}
    model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5).fit(X, Y)
    return model.predict(X1)