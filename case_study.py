import os
import argparse
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
from matplotlib.ticker import MaxNLocator
import vimpy
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from joblib import Parallel, delayed

def data_process(data_name, response, file, data_path, output_path, rep = 0, standardy = False, seed0 = None):
    df = pd.read_csv(f"{data_path}/{file}")
    tar = response
    Y_reg = df[tar]
    X_reg = df.drop(columns=[tar])
    Y_reg = df[tar]
    X_reg = df.drop(columns=[tar])

    X_train, X_test, y_train, y_test = train_test_split(X_reg, Y_reg, test_size=0.3, random_state=(seed0+rep*123))
    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    col = X_train.columns
    X = X_train.to_numpy()
    X1 = X_test.to_numpy()

    if standardy:
        scalery = StandardScaler()
        y_train = scalery.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test = scalery.transform(y_test.values.reshape(-1, 1)).ravel()
        Y = y_train
        Y1 = y_test
    else:
        Y = y_train.to_numpy()
        Y1 = y_test.to_numpy()

    training_data = pd.DataFrame(X_train)
    training_data[tar] = y_train
    training_data.to_csv(f"{output_path}/{data_name}_train_{rep}.csv", index=False)
    test_data = pd.DataFrame(X_test)
    test_data[tar] = y_test
    test_data.to_csv(f"{output_path}/{data_name}_test_{rep}.csv", index=False)
    
    data = {}
    data['col'] = col
    data['X'] = X
    data['X1'] = X1
    data['Y'] = Y
    data['Y1'] = Y1

    f = open(f'{output_path}/{data_name}_{rep}.pkl','wb')
    pickle.dump(data,f)
    f.close()


### VIM
def cs_vim(data_name, ml_name, alpha, output_path, rep = 0, measure_type='r_squared', selected_features=[], seed0 = None):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    Y = data['Y']
    col = data['col']
    ml_model = get_model_split(ml_name)
    N = len(X)
    M=len(X[0])

    np.random.seed(seed0+1234+rep*123)
    folds_outer = np.random.choice(a = np.arange(2), size = N, replace = True, p = np.array([0.5, 0.5]))
    x_1=X[folds_outer==1,:]
    y_1=Y[np.ix_(folds_outer==1)]
    x_0 = X[folds_outer==0,:]
    y_0 = Y[folds_outer==0]
    full_fit = np.array(ml_model().fit(x_1,y_1).predict(x_1))

    res=[]
    if len(selected_features)==0:
        selected_features = range(M)
    for i in selected_features:
        x_small = np.delete(x_0, i, 1) # delete the columns in s
        small_fit = np.array(ml_model().fit(x_small,y_0).predict(x_small))
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

    df_vim = pd.DataFrame(res, columns=['vim', 'pval', 'lb', 'up'])
    df_vim['feature'] = col
    df_vim = df_vim.sort_values(by='pval', ascending=True)
    file_path = f'{output_path}/{data_name}_VIM_{ml_name}_{rep}.csv'
    df_vim.to_csv(file_path, index=False)

### CPI
def cs_cpi(data_name, ml_name, response, alpha, output_path, rep = 0, seed0=None):
    
    r_code = f"""
    library(cpi)
    library(mlr3)
    library(mlr3learners)
    library(methods)
    library(glmnet)
    library(randomForest)
    library(rpart)

    train = read.csv("{output_path}/{data_name}_train_{rep}.csv")
    tar = "{response}"

    alpha = {alpha}
    p = ncol(train) - 1
    seed_ind = {rep}*4321+{seed0}

    if ("{ml_name}" == "RandomForest") {{
        ml_model = lrn(
            "regr.ranger",
            num.trees = 200,
            mtry = floor(p / 3),
            min.node.size = 5,
            replace = FALSE,
            seed = 242+seed_ind
        )
    }}
    if ("{ml_name}" == "Ridge") {{
        X = as.matrix(train[, setdiff(colnames(train), tar)])
        y = train[[tar]]
        lambda_seq = 10^seq(-3, 1, length.out = 50)
        set.seed(399+seed_ind)
        cv_fit = cv.glmnet(
            x = X,
            y = y,
            alpha = 0,
            lambda = lambda_seq,
            nfolds = 5,
            intercept = FALSE,
            standardize = FALSE
            )
            best_lambda = cv_fit$lambda.min
        ml_model = lrn(
            "regr.glmnet",
            alpha = 0,
            lambda = best_lambda,
            intercept = FALSE,
            standardize = FALSE
        )
    }}
    set.seed(432+seed_ind)
    task = as_task_regr(x = train, target = tar)
    cpi_lm_log = cpi(
        task = as_task_regr(train, target = tar),
        learner = ml_model,
        alpha = alpha,
        resampling = rsmp("holdout"),
        test = "t",
        measure = 'regr.mse',
        log = FALSE
    )

    cpi_lm_log$lb = cpi_lm_log$CPI - qnorm(1 - alpha/2) * cpi_lm_log$SE
    cpi_lm_log$ub = cpi_lm_log$CPI + qnorm(1 - alpha/2) * cpi_lm_log$SE

    cpi_new = cpi_lm_log[, c("Variable", "CPI", "estimate", "p.value", "lb", "ub")]
    names(cpi_new) <- c("feature", "cpi", "estimate", "pval", "lb", "ub")
    cpi = cpi_new[order(cpi_new$pval), ]
    
    write.csv(cpi, "{output_path}/{data_name}_CPI_{ml_name}_{rep}.csv")
    print("CPI Done")
    """

    robjects.r(r_code)


### LOCO-Split
def cs_locosplit(data_name, ml_name, ratios, alpha, bonf, output_path, rep = 0, seed0=None):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    Y = data['Y']
    X1 = data['X1']
    Y1 = data['Y1']
    col = data['col']
    M=len(X[0])
    ml_model = get_model_split(ml_name)
    for ratio in ratios:
        ress = {}
        np.random.seed(seed0+int(rep*37+ratio*100))
        result = LOCOSplit(X, Y, X1, Y1, ml_model, ratio, alpha=alpha, lossfunc="sq", selected_features=[], bonf=bonf)
        ress['inf'] = result['inf']
        ress['err2'] = result['err2']
        ress['looerr'] = result['resids_LOO']
        ress['z'] = result['z']
        f = open(f'{output_path}/{data_name}_res_LOCO-Split{ratio}_{ml_name}_{rep}.pkl','wb')
        pickle.dump(ress,f)
        f.close()

        res = pd.read_pickle(f"{output_path}/{data_name}_res_LOCO-Split{ratio}_{ml_name}_{rep}.pkl")
        print(f"LOCO-Split{ratio} loo error", res['looerr'].mean())
        err = {}
        err['err'] = res['err2'][0].mean()
        f = open(f'{output_path}/{data_name}_prederr_LOCO-Split{ratio}_{ml_name}_{rep}.pkl','wb')
        pickle.dump(err,f)
        f.close()

        zz = [res['z'][i].mean() for i in range(M)]
        locosplit = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
        locosplit['ci_center'] = 0.5*(locosplit['lb'] + locosplit['ub'])
        locosplit['ci_err'] = 0.5*(locosplit['ub'] - locosplit['lb'])
        locosplit = locosplit.sort_values(by='pval', ascending=True)
        file_path = f'{output_path}/{data_name}_LOCO-Split{ratio}_{ml_name}_{rep}.csv'
        locosplit.to_csv(file_path, index=False)

### LOCO-AdaMP
def cs_locoadamp(data_name, ml_name, n, m, alpha, indep_delta, bonf, output_path, rep = 0, seed0=None):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    Y = data['Y']
    X1 = data['X1']
    Y1 = data['Y1']
    M=len(X[0])
    Kb = [1000, 1000, 1000, 1000, 10000]

    ml_model = get_model(ml_name)
    np.random.seed(seed0+int(rep*53+n*12+m*123))
    indep_samp_prob = np.array([m/M] * M)
    res = LOCOAdaMP(X, Y, X1, Y1, n, m, Kb, ml_model, indep_samp_prob, alpha, lossfunc="sq", indep_delta=indep_delta, selected_features = [], bonf=bonf)   
    ress = {}        
    ress['inf'] = res[4]['inf']
    ress['err2'] = res[4]['err2']
    ress['z'] = res[4]['z']
    f = open(f'{output_path}/{data_name}_res_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl','wb')
    pickle.dump(ress,f)
    f.close() 
    loo_adamp = {}
    loo_adamp['looerr'] = res[4]['resids_LOO'].mean()
    f = open(f'{output_path}/{data_name}_looerr_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl','wb')
    pickle.dump(loo_adamp,f)
    f.close()

    ress_ite = {}
    for ite in range(len(Kb)):
        ress_ite[ite] = {}
        ress_ite[ite]['samp_prob'] = res[ite]['indep_samp_prob']
        ress_ite[ite]['loo_err'] = res[ite]['resids_LOO']
        ress_ite[ite]['pred_err'] = res[ite]['err2']
        f = open(f'{output_path}/{data_name}_resite_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl','wb')
        pickle.dump(ress_ite,f)
        f.close() 

def cs_locomp(data_name, ml_name, n, m, alpha, bonf, output_path, rep = 0, seed0=None):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    Y = data['Y']
    X1 = data['X1']
    Y1 = data['Y1']
    Kb = [1000, 1000, 1000, 1000, 10000]
    K_locomp = sum(Kb)

    ml_model = get_model(ml_name)
    ress = {}
    np.random.seed(seed0+int(rep*35+n*456+m*321))
    res = LOCOMP(X,Y,X1,Y1, n, m, K_locomp, ml_model, alpha=alpha, lossfunc="sq", selected_features=[], bonf=bonf)
    ress['inf'] = res['inf']
    ress['err2'] = res['err2']
    ress['z'] = res['z']
    loo_mp = res['resids_LOO'].mean()
    f = open(f'{output_path}/{data_name}_res_LOCO-MP_{ml_name}_n{n}_m{m}_{rep}.pkl','wb')
    pickle.dump(ress,f)
    f.close()
    loo_mp = {}
    loo_mp['looerr'] = res['resids_LOO'].mean()
    f = open(f'{output_path}/{data_name}_looerr_LOCO-MP_{ml_name}_n{n}_m{m}_{rep}.pkl','wb')
    pickle.dump(loo_mp,f)
    f.close()

def locoadamp_res(data_name, ml_name, n, m, output_path, figure_path, rep = 0):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    col = data['col']
    M=len(X[0])
    res = pd.read_pickle(f"{output_path}/{data_name}_res_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl")
    err = {}
    err['err'] = res['err2'][0].mean()
    f = open(f'{output_path}/{data_name}_prederr_LOCO-AdaMP_{ml_name}_{rep}.pkl','wb')
    pickle.dump(err,f)
    f.close()
    zz = [res['z'][i].mean() for i in range(M)]
    locoadamp = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
    locoadamp['ci_center'] = 0.5*(locoadamp['lb'] + locoadamp['ub'])
    locoadamp['ci_err'] = 0.5*(locoadamp['ub'] - locoadamp['lb'])
    locoadamp = locoadamp.sort_values(by='pval', ascending=True)
    file_path = f'{output_path}/{data_name}_LOCO-AdaMP_{ml_name}_{rep}.csv'
    locoadamp.to_csv(file_path, index=False)

    result  = pd.read_pickle(f"{output_path}/{data_name}_resite_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl")
    samp_prob = [0]*len(result)
    for b in range(len(result)):
        samp_prob[b] = result[b]['samp_prob']

    df_samp_prob = pd.DataFrame(samp_prob).T
    df_samp_prob['feature'] = col
    df_samp_prob = df_samp_prob.loc[locoadamp.index]
    sel_samp_prob = df_samp_prob.head(10)

    plt.figure(figsize=(5, 3))
    for b in range(len(result)):            
        plt.plot(sel_samp_prob['feature'],sel_samp_prob.iloc[:, b],linestyle='-', marker='o', markersize=4, label = f'Iteration {b+1}')
    plt.xticks(sel_samp_prob['feature'])
    plt.xlabel('Feature')
    plt.ylabel('Sampling Probability')
    plt.title(f'{ml_name}')
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    plt.tight_layout()
    plt.savefig(f'{figure_path}/{data_name}_SampProb_{ml_name}_{rep}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def locomp_res(data_name, ml_name, n, m, output_path, rep = 0):
    data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
    X = data['X']
    col = data['col']
    M=len(X[0])
    res = pd.read_pickle(f"{output_path}/{data_name}_res_LOCO-MP_{ml_name}_n{n}_m{m}_{rep}.pkl")
    err = {}
    err['err'] = res['err2'][0].mean()
    f = open(f'{output_path}/{data_name}_prederr_LOCO-MP_{ml_name}_{rep}.pkl','wb')
    pickle.dump(err,f)
    f.close()
    zz = [res['z'][i].mean() for i in range(M)]
    locomp = pd.DataFrame({'feature': col, 'pval':res['inf'][:,0], 'lb':res['inf'][:,2], 'ub':res['inf'][:,3], 'z':zz})
    locomp['ci_center'] = 0.5*(locomp['lb'] + locomp['ub'])
    locomp['ci_err'] = 0.5*(locomp['ub'] - locomp['lb'])
    locomp = locomp.sort_values(by='pval', ascending=True)
    file_path = f'{output_path}/{data_name}_LOCO-MP_{ml_name}_{rep}.csv'
    locomp.to_csv(file_path, index=False)
        
def loco_ci(data_name, ml_name, output_path, figure_path, rep = 0):
    loco_meths = ["LOCO-AdaMP","LOCO-MP","LOCO-Split0.5","LOCO-Split0.75"]
    df_ci = {}
    locoadamp = pd.read_csv(f'{output_path}/{data_name}_LOCO-AdaMP_{ml_name}_{rep}.csv')
    df_ci["LOCO-AdaMP"] = locoadamp.head(10)       
    locomp = pd.read_csv(f'{output_path}/{data_name}_LOCO-MP_{ml_name}_{rep}.csv')
    df_ci["LOCO-MP"] = locomp.head(10)
    locosplit1 = pd.read_csv(f'{output_path}/{data_name}_LOCO-Split0.5_{ml_name}_{rep}.csv')
    df_ci["LOCO-Split0.5"] = locosplit1.head(10)
    locosplit2 = pd.read_csv(f'{output_path}/{data_name}_LOCO-Split0.75_{ml_name}_{rep}.csv')
    df_ci["LOCO-Split0.75"] = locosplit2.head(10)

    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    ax = ax.flatten()   
    for l, loco_meth in enumerate(loco_meths):
        ci = df_ci[f'{loco_meth}']
        ax[l].errorbar(x=ci['feature'], y=ci['ci_center'], yerr=ci['ci_err'], linestyle="None", capsize=3, marker="_")
        ax[l].axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
        ax[l].set_xticks(ci['feature'])
        ax[l].set_xticklabels(ci['feature'], rotation=15)
        ax[l].set_xlabel('Feature')
        ax[l].set_ylabel('90% Confidence Interval')
        ax[l].set_title(f'{loco_meth} | {ml_name}')
        ax[l].grid(True, zorder=0, alpha=0.5, linestyle='--')           
    #fig.delaxes(ax[4])
    plt.tight_layout()
    plt.savefig(f'{figure_path}/{data_name}_CI_{ml_name}_{rep}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()


def loco_pred_error_res(data_name, ml_name, output_path, figure_path, rep = 0):
    loco_meths = ["AdaMP","MP","Split0.5","Split0.75"]
    pred_err = [0]*4

    adamp_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-AdaMP_{ml_name}_{rep}.pkl')
    pred_err[0] = adamp_err['err']
    mp_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-MP_{ml_name}_{rep}.pkl')
    pred_err[1] = mp_err['err']
    split1_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-Split0.5_{ml_name}_{rep}.pkl')
    pred_err[2] = split1_err['err']
    split2_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-Split0.75_{ml_name}_{rep}.pkl')
    pred_err[3] = split2_err['err']

    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd', "#8c564b"]
    plt.figure(figsize=(4, 3))
    plt.bar(loco_meths, pred_err,  color=color_list)
    plt.title(f"{ml_name}")
    plt.xlabel("Method")
    #plt.xticks(rotation=15)
    plt.ylabel("Test Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(f'{figure_path}/{data_name}_PredError_{ml_name}_{rep}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()


def loco_pred_error(data_name, ml_name, output_path, figure_path, reps = [0]):
    loco_meths = ["AdaMP","MP","Split0.5","Split0.75"]
    pred_err = [0]*4

    for rep in reps:
        adamp_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-AdaMP_{ml_name}_{rep}.pkl')
        pred_err[0] = pred_err[0] + adamp_err['err']/len(reps)
        mp_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-MP_{ml_name}_{rep}.pkl')
        pred_err[1] = pred_err[1] + mp_err['err']/len(reps)
        split1_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-Split0.5_{ml_name}_{rep}.pkl')
        pred_err[2] = pred_err[2] + split1_err['err']/len(reps)
        split2_err = pd.read_pickle(f'{output_path}/{data_name}_prederr_LOCO-Split0.75_{ml_name}_{rep}.pkl')
        pred_err[3] = pred_err[3] + split2_err['err']/len(reps)

    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd', "#8c564b"]
    plt.figure(figsize=(4, 3))
    plt.bar(loco_meths, pred_err,  color=color_list)
    plt.title(f"{ml_name}")
    plt.xlabel("LOCO Method")
    #plt.xticks(rotation=15)
    plt.ylabel("Test Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(f'{figure_path}/{data_name}_PredError_{ml_name}_mean{len(reps)}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def tmse_res(data_name, response, ml_name, output_path, figure_path, rep, seed0):
    train = pd.read_csv(f"{output_path}/{data_name}_train_{rep}.csv")
    test = pd.read_csv(f"{output_path}/{data_name}_test_{rep}.csv")
    y_train = train[response]
    X_train = train.drop(columns=[response])
    y_test = test[response]
    X_test = test.drop(columns=[response])

    adamp_mse = [0]*50
    mp_mse = [0]*50
    split1_mse = [0]*50
    split2_mse = [0]*50
    cpi_mse = [0]*50
    vim_mse = [0]*50

    locoadamp = pd.read_csv(f"{output_path}/{data_name}_LOCO-AdaMP_{ml_name}_{rep}.csv")
    locoadamp = locoadamp.sort_values(by='pval', ascending=True)

    locomp = pd.read_csv(f"{output_path}/{data_name}_LOCO-MP_{ml_name}_{rep}.csv")
    locomp = locomp.sort_values(by='pval', ascending=True)

    locosplit1 = pd.read_csv(f"{output_path}/{data_name}_LOCO-Split0.5_{ml_name}_{rep}.csv")
    locosplit1 = locosplit1.sort_values(by='pval', ascending=True)

    locosplit2 = pd.read_csv(f"{output_path}/{data_name}_LOCO-Split0.75_{ml_name}_{rep}.csv")
    locosplit2 = locosplit2.sort_values(by='pval', ascending=True)

    cpi = pd.read_csv(f"{output_path}/{data_name}_CPI_{ml_name}_{rep}.csv")
    cpi = cpi.sort_values(by='pval', ascending=True)

    vim = pd.read_csv(f"{output_path}/{data_name}_VIM_{ml_name}_{rep}.csv")
    vim = vim.sort_values(by='pval', ascending=True)

    for i in range(50):
        seed = i*321+5432+seed0
        adamp_fea = locoadamp['feature'][0:i+1]
        adamp_fea1 = adamp_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[adamp_fea1],y_train)
        y_pred = rf.predict(X_test[adamp_fea1])
        adamp_mse[i] = mean_squared_error(y_test, y_pred)
  
        mp_fea = locomp['feature'][0:i+1]
        mp_fea1 = mp_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[mp_fea1],y_train)
        y_pred = rf.predict(X_test[mp_fea1])
        mp_mse[i] = mean_squared_error(y_test, y_pred)
   
        split1_fea = locosplit1['feature'][0:i+1]
        split1_fea1 = split1_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[split1_fea1],y_train)
        y_pred = rf.predict(X_test[split1_fea1])
        split1_mse[i] = mean_squared_error(y_test, y_pred)

        split2_fea = locosplit2['feature'][0:i+1]
        split2_fea1 = split2_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[split2_fea1],y_train)
        y_pred = rf.predict(X_test[split2_fea1])
        split2_mse[i] = mean_squared_error(y_test, y_pred)

        cpi_fea = cpi['feature'][0:i+1]
        cpi_fea1 = cpi_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[cpi_fea1],y_train)
        y_pred = rf.predict(X_test[cpi_fea1])
        cpi_mse[i] = mean_squared_error(y_test, y_pred)

        vim_fea = vim['feature'][0:i+1]
        vim_fea1 = vim_fea.sort_values(key=lambda s: s.str.extract(r'X(\d+)')[0].astype(int))
        rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=seed, bootstrap=False)
        rf.fit(X_train[vim_fea1],y_train)
        y_pred = rf.predict(X_test[vim_fea1])
        vim_mse[i] = mean_squared_error(y_test, y_pred)

    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', "#8c564b", '#9467bd']

    x = np.arange(1, 51)
    plt.figure(figsize=(4, 3))  
    plt.plot(x, adamp_mse, marker='o', markersize=4, label='LOCO-AdaMP', color = color_list[0], zorder=10)
    plt.plot(x, mp_mse, marker='o', markersize=4, label='LOCO-MP', color = color_list[1], zorder=9) 
    plt.plot(x, split1_mse, marker='o', markersize=4, label='LOCO-Split0.5', color = color_list[2], zorder=8)
    plt.plot(x, split2_mse, marker='o', markersize=4, label='LOCO-Split0.75', color = color_list[3], zorder=7) 
    plt.plot(x, cpi_mse, marker='o', markersize=4, label='CPI', color = color_list[4], zorder=6)
    plt.plot(x, vim_mse, marker='o', markersize=4, label='VIM', color = color_list[5], zorder=5)

    plt.title(f"{ml_name}")
    plt.xlabel("Number of Features")
    plt.ylabel("Test Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{figure_path}/{data_name}_TMSE_{ml_name}_{rep}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()
    
    tmse = pd.DataFrame({'LOCO-AdaMP': adamp_mse, 'LOCO-MP': mp_mse, 'LOCO-Split0.5': split1_mse, 'LOCO-Split0.75': split2_mse, 'CPI': cpi_mse, 'VIM': vim_mse})
    f = open(f'{output_path}/{data_name}_TMSE_{ml_name}_{rep}.pkl','wb')
    pickle.dump(tmse,f)
    f.close()

def tmse(data_name, ml_name, output_path, figure_path, reps = [0]):
    tmses = []

    for rep in reps:
        tmse = pd.read_pickle(f'{output_path}/{data_name}_TMSE_{ml_name}_{rep}.pkl')
        tmses.append(tmse)
    tmse_array = np.array([df.values for df in tmses])
    tmse_mean = pd.DataFrame(tmse_array.mean(axis=0), columns=tmses[0].columns)
    tmse_sem = pd.DataFrame(tmse_array.std(axis=0, ddof=1)/np.sqrt(tmse_array.shape[0]), columns=tmses[0].columns)
    tmse_mean_20 = tmse_mean.iloc[:20]
    tmse_sem_20 = tmse_sem.iloc[:20]

    color_list = ['#d62728', '#2ca02c','#ff7f0e','#1f77b4', "#8c564b", '#9467bd']

    x = np.arange(1, 51)
    plt.figure(figsize=(4, 3))  
    plt.plot(x, tmse_mean['LOCO-AdaMP'], marker='o', markersize=4, label='LOCO-AdaMP', color = color_list[0], zorder=10)
    plt.plot(x, tmse_mean['LOCO-MP'], marker='o', markersize=4, label='LOCO-MP', color = color_list[1], zorder=9) 
    plt.plot(x, tmse_mean['LOCO-Split0.5'], marker='o', markersize=4, label='LOCO-Split0.5', color = color_list[2], zorder=8)
    plt.plot(x, tmse_mean['LOCO-Split0.75'], marker='o', markersize=4, label='LOCO-Split0.75', color = color_list[3], zorder=7) 
    plt.plot(x, tmse_mean['CPI'], marker='o', markersize=4, label='CPI', color = color_list[4], zorder=6)
    plt.plot(x, tmse_mean['VIM'], marker='o', markersize=4, label='VIM', color = color_list[5], zorder=5)
    plt.title(f"{ml_name}")
    plt.xlabel("Number of Features")
    plt.ylabel("Test Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{figure_path}/{data_name}_TMSE_{ml_name}_mean{len(reps)}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

    x = np.arange(1, 21)
    plt.figure(figsize=(4, 3))  
    plt.errorbar(x, tmse_mean_20['LOCO-AdaMP'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['LOCO-AdaMP'], label='LOCO-AdaMP', marker='o', markersize=4, capsize=3, color = color_list[0], zorder=10)
    plt.errorbar(x, tmse_mean_20['LOCO-MP'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['LOCO-MP'], label='LOCO-MP', marker='o', markersize=4, capsize=3, color = color_list[1], zorder=9)
    plt.errorbar(x, tmse_mean_20['LOCO-Split0.5'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['LOCO-Split0.5'], label='LOCO-Split0.5', marker='o', markersize=4, capsize=3, color = color_list[2], zorder=8)
    plt.errorbar(x, tmse_mean_20['LOCO-Split0.75'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['LOCO-Split0.75'], label='LOCO-Split0.75', marker='o', markersize=4, capsize=3, color = color_list[3], zorder=7)
    plt.errorbar(x, tmse_mean_20['CPI'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['CPI'], label='CPI', marker='o', markersize=4, capsize=3, color = color_list[4], zorder=6)
    plt.errorbar(x, tmse_mean_20['VIM'], yerr=(norm.ppf(1-alpha/2))*tmse_sem_20['VIM'], label='VIM', marker='o', markersize=4, capsize=3, color = color_list[5], zorder=5)
    plt.title(f"{ml_name}")
    plt.xlabel("Number of Features")
    plt.ylabel("Test Error")
    plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{figure_path}/{data_name}_TMSECI_{ml_name}_mean{len(reps)}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def num_sig(data_name, ml_name, col, output_path, figure_path, reps = [0]):
    sig_counts_df = pd.DataFrame({'feature': col})
    methods = ["LOCO-AdaMP", "LOCO-MP", "LOCO-Split0.5", "LOCO-Split0.75", "CPI", "VIM"]
    for method in methods:
        sig_counts = {feature: 0 for feature in col}   
        for rep in reps:
            file_path = f"{output_path}/{data_name}_{method}_{ml_name}_{rep}.csv"
            df = pd.read_csv(file_path)       
            for idx, row in df.iterrows():
                feature = row['feature']
                pval = row['pval']
                if feature in sig_counts and pval < 0.1: 
                    sig_counts[feature] += 1              
        sig_counts_df[method] = sig_counts_df['feature'].map(sig_counts)
    csv_path = os.path.join(figure_path, f"{data_name}_sig_counts_all_methods.csv")
    sig_counts_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(11,6))
    axes = axes.flatten()
    max_count = 0
    for method in methods:
        counts = sig_counts_df[method].values
        hist, _ = np.histogram(counts, bins=np.arange(-0.5, 11.5, 1))
        max_count = max(max_count, hist.max())

    for i, method in enumerate(methods):
        counts = sig_counts_df[method].values
    
        n_hist, bins, patches = axes[i].hist(counts, bins=np.arange(-0.5, 11.5, 1), rwidth=0.8)

        for j, patch in enumerate(patches):
            if j == 0:
                patch.set_facecolor('#1f77b4')
            else:
                patch.set_facecolor('#ff7f0e')
        axes[i].set_title(method)
        axes[i].set_xlabel('Selection Frequency')
        axes[i].set_ylabel('Number of Features')
        axes[i].set_xticks(range(0,11))
        axes[i].set_ylim(0, max_count+5)
        axes[i].grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/{data_name}_Hist0_{ml_name}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7.2,3.1))
    axes = axes.flatten()
    loco2 = ["LOCO-AdaMP", "LOCO-MP"]
    max_count = 0

    for method in loco2:
        counts = sig_counts_df[method].values
        counts = counts[counts > 0]
        hist, _ = np.histogram(counts, bins=np.arange(-0.5, 11.5, 1))
        max_count = max(max_count, hist.max())

    for i, method in enumerate(loco2):
        counts = sig_counts_df[method]
        counts = counts[counts > 0]
        axes[i].hist(counts, bins=np.arange(0.5, 11.5, 1), color = '#ff7f0e', rwidth=0.8)
        axes[i].set_title(method)
        axes[i].set_xlabel('Selection Frequency')
        axes[i].set_ylabel('Number of Features')
        axes[i].set_xticks(range(1,11))
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].set_ylim(0, max_count+0.25)
        axes[i].grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/{data_name}_Hist_{ml_name}.png', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()


def get_results(data_name, response, ml_name, ms, ns, output_path, figure_path, rep, seed0):
    looerr_adamp = [0]*len(ns)
    looerr_mp = [0]*len(ns)
    for i,n in enumerate(ns):
        looerr_adamp[i] = [0]*len(ms)
        looerr_mp[i] = [0]*len(ms)
        for j,m in enumerate(ms):
            adamp = pd.read_pickle(f'{output_path}/{data_name}_looerr_LOCO-AdaMP_{ml_name}_n{n}_m{m}_{rep}.pkl')
            mp = pd.read_pickle(f'{output_path}/{data_name}_looerr_LOCO-MP_{ml_name}_n{n}_m{m}_{rep}.pkl')
            looerr_adamp[i][j] = adamp['looerr']
            looerr_mp[i][j] = mp['looerr']

    loo_err_adamp = pd.DataFrame(looerr_adamp)
    loo_err_adamp.columns = [f"m={i}" for i in ms]
    loo_err_adamp.index = [f"n={i}" for i in ns]
    print(loo_err_adamp)
    min_value = loo_err_adamp.values.min()
    row_idx_adamp, col_idx_adamp = np.where(loo_err_adamp.values == min_value)
    print('LOCO-AdaMP:', 'm=', ms[col_idx_adamp[0]], ',n=',ns[row_idx_adamp[0]])
    n_adamp = ns[row_idx_adamp[0]]
    m_adamp = ms[col_idx_adamp[0]]

    loo_err_mp = pd.DataFrame(looerr_mp)
    loo_err_mp.columns = [f"m={i}" for i in ms]
    loo_err_mp.index = [f"n={i}" for i in ns]
    print(loo_err_mp)
    min_value = loo_err_mp.values.min()
    row_idx_mp, col_idx_mp = np.where(loo_err_mp.values == min_value)
    print('LOCO-AdaMP:', 'm=', ms[col_idx_mp[0]], ',n=',ns[row_idx_mp[0]])
    n_mp = ns[row_idx_mp[0]]
    m_mp = ms[col_idx_mp[0]]

    locoadamp_res(data_name, ml_name, n_adamp, m_adamp, output_path, figure_path, rep = rep)
    locomp_res(data_name, ml_name, n_mp, m_mp, output_path, rep = rep)
    loco_ci(data_name, ml_name, output_path, figure_path, rep = rep)
    loco_pred_error_res(data_name, ml_name, output_path, figure_path, rep = rep)
    tmse_res(data_name, response, ml_name, output_path, figure_path, rep=rep, seed0 = seed0)

def get_figures(data_name, ml_name, output_path, figure_path, reps = [0]):
    loco_pred_error(data_name, ml_name, output_path, figure_path, reps = reps)
    tmse(data_name, ml_name, output_path, figure_path, reps = reps)


if __name__ == "__main__":
    ### Parameter setting
    data_path = "Real_Data"
    if not os.path.exists(data_path):
        print("No Data Found")

    output = "Output_Data"
    if not os.path.exists(output):
        os.mkdir(output)

    output_path = "Output_Data/Case_Study"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    figure = "Figure"
    if not os.path.exists(figure):
        os.mkdir(figure)

    figure_path = "Figure/Case_Study"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    plt.rcParams['axes.titlesize'] = 12.5
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9.5

    data_name = 'rosmap86'
    file = f'{data_name}.csv'
    ml_name = "RandomForest"
    replicate = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--ml", type=str, default="randomforest", choices=["ridge", "randomforest"])
    parser.add_argument("--rep", type=int, default=0)

    args = parser.parse_args()
    data_name = args.data
    ml_name = format_ml_name(args.ml)
    replicate = args.rep

    seed0 = 101

    alpha = 0.1
    indep_delta = 0.9
    ratios = [0.5, 0.75]
    response = "y"
    file = f'{data_name}.csv'
    bonf = True
    standy = False
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    reps = range(1, replicate + 1)
    if replicate == 0:
        reps = [0]


    def run_func(data_name, ml_name, response, file, ratios, alpha, indep_delta, data_path, output_path, figure_path, bonf, standy, seed0 = None, rep = 1):
        seed0 = get_seed0(seed0)
        check_path = f'{output_path}/{data_name}_{rep}.pkl'
        if not os.path.exists(check_path):
            data_process(data_name, response, file, data_path, output_path, rep = rep, standardy = standy, seed0 = seed0)
        
        cs_vim(data_name, ml_name, alpha, output_path, rep = rep, seed0 = seed0)
        cs_cpi(data_name, ml_name, response, alpha, output_path, rep = rep, seed0 = seed0)
        cs_locosplit(data_name, ml_name, ratios, alpha, bonf, output_path, rep = rep, seed0 = seed0)
        data = pd.read_pickle(f'{output_path}/{data_name}_{rep}.pkl')
        X = data['X']
        N = X.shape[0]
        M = X.shape[1]
        m0, n0 = get_mn(M,N)
        ns = [n0]
        ms = [int(0.05*M), int(0.1*M), int(0.2*M), int(0.4*M)]
        for m in ms:
            for n in ns:
                cs_locoadamp(data_name, ml_name, n, m, alpha, indep_delta, bonf, output_path, rep = rep, seed0 = seed0)
                cs_locomp(data_name, ml_name, n, m, alpha, bonf, output_path, rep = rep, seed0 = seed0)
        get_results(data_name, response, ml_name, ms, ns, output_path, figure_path, rep = rep, seed0 = seed0)

    Parallel(n_jobs=n_jobs)(delayed(run_func)(data_name, ml_name, response, file, ratios, alpha, indep_delta, data_path, output_path, figure_path, bonf, standy, seed0, rep) for rep in reps)

    data = pd.read_pickle(f'{output_path}/{data_name}_{replicate}.pkl')
    col = data['col']
    if len(reps) > 1:
        get_figures(data_name, ml_name, output_path, figure_path, reps = reps)
        num_sig(data_name, ml_name, col, output_path, figure_path, reps = reps)
    print("--Done--")
