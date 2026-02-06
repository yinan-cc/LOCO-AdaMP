import os
import numpy as np
import pandas as pd
import pickle
from loco_methods import *
from ml_models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils import *
from scipy.stats import *
import matplotlib.pyplot as plt
import vimpy

data_name = 'rosmap86'
df = pd.read_csv('rosmap_86.csv').drop(columns=['Unnamed: 0', "y_class"])
tar = "y_reg"
Y_reg = df[tar]
X_reg = df.drop(columns=[tar])
k = [100,100,100,100,1000]


Y_reg = df[tar]
X_reg = df.drop(columns=[tar])
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, Y_reg,
    test_size=0.3,
    random_state=42
)
scaler = StandardScaler().set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

col = X_train.columns
X = X_train.to_numpy()
X1 = X_test.to_numpy()
Y = y_train.to_numpy()
Y1 = y_test.to_numpy()

training_data = pd.DataFrame(X_train)
training_data[tar] = y_train
training_data.to_csv(f"{data_name}_train.csv", index=False)
test_data = pd.DataFrame(X_test)
test_data[tar] = y_test
test_data.to_csv(f"{data_name}_test.csv", index=False)

out = "Output_Data"
if not os.path.exists(out):
    os.mkdir(out)
output = "Output_Data/Case_Study"
if not os.path.exists(output):
    os.mkdir(output)
fig = "Figure"
if not os.path.exists(fig):
    os.mkdir(fig)
figure = "Figure/Case_Study"
if not os.path.exists(figure):
    os.mkdir(figure)

N = X.shape[0]
M = X.shape[1]

m_,n_ = get_mn(M,N)
Kb_ = get_Kb(M, N, m_, n_, k)
print(Kb_)

m0, n0 = get_mn(M,N)
ns = [n0]
ms = [int(0.1*M), int(0.2*M), int(0.3*M)]

alpha = 0.1
indep_delta = 0.9
ratios = [0.5,0.75] # for LOCO-Split
loco_meths = ["LOCO-AdaMP","LOCO-MP","LOCO-Split0.5","LOCO-Split0.75"]
ml_names = ["RandomForest"]#ml_names = ["Ridge", "RandomForest", "KernelRidge"]
pred_err = [0]*4

### VIM
def vimee(X, Y, ml_model, alpha, measure_type='r_squared', selected_features=[]):
    N = len(X)
    M=len(X[0])
    np.random.seed(100)
    folds_outer = np.random.choice(a = np.arange(2), size = N, replace = True, p = np.array([0.5, 0.5]))
    ## fit the full regression
    # cv_full.fit(x[folds_outer == 1, :], y[folds_outer == 1])
    # full_fit = cv_full.best_estimator_.predict(x[folds_outer == 1, :])
    x_1=X[folds_outer==1,:]
    y_1=Y[np.ix_(folds_outer==1)]
    x_0 = X[folds_outer==0,:]
    y_0 = Y[folds_outer==0]
    ## prediction on x1 use x1
    full_fit = np.array(ml_model(x_1,y_1,x_1))
    res=[]

    if len(selected_features)==0:
        selected_features = range(M)

    for i in selected_features:
        x_small = np.delete(x_0, i, 1) # delete the columns in s
        small_fit = np.array(ml_model(x_small,y_0,x_small))
        vimp_precompute = vimpy.vim(y = Y, x = X, s = i, f = full_fit, r = small_fit, measure_type = measure_type, folds = folds_outer)
        vimp_precompute.get_point_est()
            ## get the influence function estimate
        vimp_precompute.get_influence_function()
        ## get a standard error
        vimp_precompute.get_se()
        ## get a confidence interval
        vimp_precompute.get_ci(level=0.9)
        vimp_precompute.hypothesis_test(alpha = alpha, delta = 0)

        res.append([vimp_precompute.vimp_]+[vimp_precompute.p_value_]+list(vimp_precompute.ci_[0]))
    return res

for ml_name in ml_names:
    ml_model = get_model_split(ml_name)
    res = vimee(X, Y, ml_model, alpha = 0.1, measure_type='r_squared')
    vim = pd.DataFrame(res, columns=['vim', 'pval', 'lb', 'up'])
    vim['feature'] = col
    vim = vim.sort_values(by='pval', ascending=True)
    file_path = f'{output}/{data_name}_VIM_{ml_name}.csv'
    vim.to_csv(file_path, index=False)

### LOCO-Split
for ml_name in ml_names:
    ml_model = get_model_split(ml_name)
    for ratio in ratios:
        ress = {}            
        res = LOCOSplit(X, Y, X1, Y1, ml_model, ratio, alpha=alpha, selected_features=[], bonf=False)
        ress['inf'] = res['inf']
        ress['err2'] = res['err2_sq']
        ress['looerr'] = res['resids_LOO_sq']
        ress['z'] = res['z']
        f = open(f'{output}/{data_name}_res_LOCO-Split{ratio}_{ml_name}.pkl','wb')
        pickle.dump(ress,f)
        f.close()

### LOCO-Split Results
res = pd.read_pickle(f"{output}/{data_name}_res_LOCO-Split0.5_RandomForest.pkl")
print(res['looerr'].mean())
pred_err[2] = res['err2'][0].mean()
zz = [res['z'][i].mean() for i in range(M)]
locosplit1 = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
locosplit1['ci_center'] = 0.5*(locosplit1['lb'] + locosplit1['ub'])
locosplit1['ci_err'] = 0.5*(locosplit1['ub'] - locosplit1['lb'])
locosplit1 = locosplit1.sort_values(by='pval', ascending=True)
file_path = f'{output}/{data_name}_LOCO-Split0.5_RandomForest.csv'
locosplit1.to_csv(file_path, index=False)
sel_split1 = locosplit1[locosplit1['lb']>0]
if (sel_split1.shape[0] != 0):
    if (sel_split1.shape[0] > 5):
        sel_split1 = sel_split1.head()
    
    plt.figure(figsize=(4, 3))
    plt.errorbar(x=sel_split1['feature'], y=sel_split1['ci_center'], yerr=sel_split1['ci_err'],linestyle="None", capsize=3, marker="_")
    plt.title("Random Forest")
    plt.xlabel("Feature")
    plt.xticks(sel_split1['feature'], rotation = 15)
    plt.ylabel("90% Confidence Interval")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(f'{figure}/{data_name}_CI_LOCO-Split0.5_RandomForest.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

res = pd.read_pickle(f"{output}/{data_name}_res_LOCO-Split0.75_RandomForest.pkl")
print(res['looerr'].mean())
pred_err[3] = res['err2'][0].mean()
zz = [res['z'][i].mean() for i in range(M)]
locosplit2 = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
locosplit2['ci_center'] = 0.5*(locosplit2['lb'] + locosplit2['ub'])
locosplit2['ci_err'] = 0.5*(locosplit2['ub'] - locosplit2['lb'])
locosplit2 = locosplit2.sort_values(by='pval', ascending=True)
file_path = f'{output}/{data_name}_LOCO-Split0.75_RandomForest.csv'
locosplit2.to_csv(file_path, index=False)

sel_split2 = locosplit2[locosplit2['lb']>0]
if (sel_split2.shape[0] != 0):
    if (sel_split2.shape[0] > 5):
        sel_split2 = sel_split2.head()
    sel_split2['ci_center'] = 0.5*(sel_split2['lb'] + sel_split2['ub'])
    sel_split2['ci_err'] = 0.5*(sel_split2['ub'] - sel_split2['lb'])
    plt.figure(figsize=(4, 3))
    plt.errorbar(x=sel_split2['feature'], y=sel_split2['ci_center'], yerr=sel_split2['ci_err'],linestyle="None", capsize=3, marker="_")
    plt.title("Random Forest")
    plt.xlabel("Feature")
    plt.xticks(sel_split2['feature'], rotation = 15)
    plt.ylabel("90% Confidence Interval")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(f'{figure}/{data_name}_CI_LOCO-Split0.75_RandomForest.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def run_loco_meth(M, N, X, Y, X1, Y1, n, m, k, ml_name, indep_delta = 0.9, alpha = 0.1, bonf = False):
    Kb = get_Kb(M, N, m, n, k)
    print(Kb)
    K_locomp = sum(Kb)
    np.random.seed(100)
    ml_model = get_model(ml_name)

    ### LOCO-Adamp
    ress = {}
    indep_samp_prob = np.array([1/M] * M)
    for epo in range(len(Kb)):
        B = Kb[epo]
        res = LOCOAdaMP(X,Y,X1,Y1, n, m, B, ml_model, indep_samp_prob, alpha=alpha, selected_features=[], bonf=bonf)
        indep_samp_prob = indep_sampling_probability(res['z'], m, indep_delta)      
    ress['inf'] = res['inf']
    ress['err2'] = res['err2_sq']
    ress['looerr'] = res['resids_LOO_sq']
    ress['z'] = res['z']
    f = open(f'{output}/{data_name}_res_LOCO-AdaMP_{ml_name}_n{n}_m{m}.pkl','wb')
    pickle.dump(ress,f)
    f.close()    
    

    ### LOCO-MP
    ress = {}
    res = LOCOMP(X,Y,X1,Y1, n, m, K_locomp, ml_model, alpha=alpha, selected_features=[], bonf=bonf)
    ress['inf'] = res['inf']
    ress['err2'] = res['err2_sq']
    ress['looerr'] = res['resids_LOO_sq']
    ress['z'] = res['z']
    f = open(f'{output}/{data_name}_res_LOCO-MP_{ml_name}_n{n}_m{m}.pkl','wb')
    pickle.dump(ress,f)
    #f.close()
    plt.close()

def get_loco_result(n, m, pred_err, ml_name):
    ### LOCO-AdaMP
    res = pd.read_pickle(f"{output}/{data_name}_res_LOCO-AdaMP_{ml_name}_n{n}_m{m}.pkl")
    loo_adamp = res['looerr'].mean()
    pred_err[0] = res['err2'][0].mean()
    zz = [res['z'][i].mean() for i in range(M)]
    locoadamp = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
    locoadamp['ci_center'] = 0.5*(locoadamp['lb'] + locoadamp['ub'])
    locoadamp['ci_err'] = 0.5*(locoadamp['ub'] - locoadamp['lb'])
    locoadamp = locoadamp.sort_values(by='pval', ascending=True)
    file_path = f'{output}/{data_name}_LOCO-AdaMP_{ml_name}_n{n}_m{m}.csv'
    locoadamp.to_csv(file_path, index=False)

    sel_adamp = locoadamp[locoadamp['lb']>0]
    if (sel_adamp.shape[0] != 0):
        if (sel_adamp.shape[0] > 5):
            sel_adamp = sel_adamp.head()
        plt.figure(figsize=(4, 3))
        plt.errorbar(x=sel_adamp['feature'], y=sel_adamp['ci_center'], yerr=sel_adamp['ci_err'],linestyle="None", capsize=3, marker="_")
        plt.title("Random Forest")
        plt.xlabel("Feature")
        plt.xticks(sel_adamp['feature'], rotation = 20)
        plt.ylabel("90% Confidence Interval")
        plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
        plt.savefig(f'{figure}/{data_name}_CI_LOCO-AdaMP_{ml_name}_n{n}_m{m}.png', dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()

    ### LOCO-MP
    res = pd.read_pickle(f"{output}/{data_name}_res_LOCO-MP_{ml_name}_n{n}_m{m}.pkl")
    loo_mp = res['looerr'].mean()
    pred_err[1] = res['err2'][0].mean()
    zz = [res['z'][i].mean() for i in range(M)]
    locomp = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
    locomp['ci_center'] = 0.5*(locomp['lb'] + locomp['ub'])
    locomp['ci_err'] = 0.5*(locomp['ub'] - locomp['lb'])
    locomp = locomp.sort_values(by='pval', ascending=True)
    file_path = f'{output}/{data_name}_LOCO-MP_{ml_name}_n{n}_m{m}.csv'
    locomp.to_csv(file_path, index=False)
    
    sel_mp = locomp[locomp['lb']>0]
    if (sel_mp.shape[0] != 0):
        if (sel_mp.shape[0] > 5):
            sel_mp = sel_mp.head()
       
        plt.figure(figsize=(4, 3))
        plt.errorbar(x=sel_mp['feature'], y=sel_mp['ci_center'], yerr=sel_mp['ci_err'],linestyle="None", capsize=3, marker="_")
        plt.title("Random Forest")
        plt.xlabel("Feature")
        plt.xticks(sel_mp['feature'], rotation = 15)
        plt.ylabel("90% Confidence Interval")
        plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
        plt.savefig(f'{figure}/{data_name}_CI_LOCO-MP_{ml_name}_n{n}_m{m}.png', dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()

    ### LOCOs Prediction Erros
    plt.rcParams['axes.titlesize'] = 12.5
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9.5

    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd']#red, green, orange, blue, purple

    plt.figure(figsize=(4, 3))

    plt.bar(loco_meths, pred_err,  color=color_list)

    plt.title("Random Forest")
    plt.xlabel("LOCO Approach")
    plt.xticks(rotation=15)
    plt.ylabel("Prediction Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(f'{figure}/{data_name}_PredError_{ml_name}_n{n}_m{m}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

    return loo_adamp, loo_mp

loo_err_adamp = np.zeros((len(ns), len(ms)))
loo_err_mp = np.zeros((len(ns), len(ms)))
for i, n in enumerate(ns):
    for j, m in enumerate(ms):
        for ml_name in ml_names:
            run_loco_meth(M, N, X, Y, X1, Y1, n, m, k, ml_name, indep_delta, alpha, bonf = False)
            loo_err_adamp[i,j], loo_err_mp[i,j] = get_loco_result(n, m, pred_err, ml_name)
loo_err_adamp = pd.DataFrame(loo_err_adamp)
loo_err_adamp.columns = [f"m={i}" for i in ms]
loo_err_adamp.index = [f"n={i}" for i in ns]
loo_err_mp = pd.DataFrame(loo_err_mp)
loo_err_mp.columns = [f"m={i}" for i in ms]
loo_err_mp.index = [f"n={i}" for i in ns]
print(loo_err_adamp)
print(loo_err_mp)
f = open(f'{output}/{data_name}_LOO_LOCO-AdaMP_{ml_name}.pkl','wb')
pickle.dump(loo_err_adamp,f)
f = open(f'{output}/{data_name}_LOO_LOCO-MP_{ml_name}.pkl','wb')
pickle.dump(loo_err_mp,f)
