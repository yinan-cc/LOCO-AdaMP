import numpy as np
import pandas as pd
import pickle
from loco_methods import *
from ml_models import *
from data_generate import *
from utils import *
from scipy.stats import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

### Run prediction functions
def inf_pred_ci(M, N, simu_type, ml_name, rep, ratios, alpha, indep_delta, n_jobs, lossfunc, simu_path, output_path,seed0=1):
    pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, lossfunc, simu_path, output_path = output_path,seed0=seed0) 
    pred_ci_LOCOMP(M, N, simu_type, ml_name, rep, alpha, n_jobs, lossfunc, simu_path = simu_path, output_path = output_path,seed0=seed0)
    pred_ci_LOCOSplit(M, simu_type, ml_name, rep, ratios, alpha, n_jobs, lossfunc, simu_path = simu_path, output_path = output_path,seed0=seed0)

def res_pred_ci(M, loco_meths, simu_type, ml_name, rep, ratios, alpha, target_inds, n_features, lossfunc, simu_path = '.', output_path = '.', seed0=1):
    pred_err = []
    pe = 0
    for rep_ind in range(rep):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3] 
        ml_model = get_model_split(ml_name)
        np.random.seed(seed0+rep_ind*233+M+get_seed(simu_type, ml_name, lossfunc)*91)
        loss_ind = loss_func(ml_model().fit(X,Y).predict(X1), Y1, func = lossfunc)
        pe = pe + loss_ind.mean()/rep
    pred_err.append(pe)
    pred_err.append(res_pred_ci_LOCOAdaMP(M, simu_type, ml_name, rep, alpha, target_inds, n_features, lossfunc, output_path = output_path))

    pred_err.append(res_pred_ci_LOCOMP(M, simu_type, ml_name, rep, n_features, lossfunc, output_path = output_path))
    for ratio in ratios:
        pred_err.append(res_pred_ci_LOCOSplit(M, simu_type, ml_name, rep, ratio, n_features, lossfunc, output_path = output_path))

    dfs_pred_error = pd.DataFrame({'loco_meth':['Full-Data']+loco_meths, 'pred_err': pred_err})
    dfs_pred_error = dfs_pred_error.set_index('loco_meth')
    file_path = f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_pred_error.to_pickle(file_path)

def fig_pred_ci(M, N, loco_meths, simu_type, ml_name, indep_delta, target_inds, lossfunc, output_path = '.'):
    dfs_ci = {loco_meth: pd.DataFrame() for loco_meth in loco_meths}
    dfs_pred_error = pd.read_pickle(f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    dfs_adap_error = result = pd.read_pickle(f'{output_path}/fig_PredCI_AdaError_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    dfs_adap_ass = pd.read_pickle(f'{output_path}/fig_PredCI_AdaAss_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    dfs_adap_samp_prob = {b: pd.DataFrame() for b in range(len(result))}
    dfs_adap_ci = {b: pd.DataFrame() for b in range(len(result))}
    dfs_adap_target = {tar: pd.DataFrame() for tar in target_inds}
    for ite in range(len(result)):
        dfs_adap_samp_prob[ite] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaSampProb_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_epoch{ite}.pkl')
        dfs_adap_ci[ite]= pd.read_pickle(f'{output_path}/fig_PredCI_AdaCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_epoch{ite}.pkl')
    for loco_meth in loco_meths:
        dfs_ci[f'{loco_meth}'] = pd.read_pickle(f'{output_path}/fig_PredCI_CI_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    for tar in target_inds:
        dfs_adap_target[tar] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaTarget{tar}_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    fig_pred_ci_LOCO(M, simu_type, ml_name, loco_meths, dfs_pred_error, dfs_ci, lossfunc, output_path = output_path)
    fig_pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci, dfs_adap_ass, lossfunc)

### Prediction for LOCO-AdaMP
def pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, lossfunc, simu_path = '.', output_path = '.', seed0=1):
    m,n,Kb,K_locomp = get_parameter(M,N)
    ml_model = get_model(ml_name)
    def get_predci_LOCOAdaMP(rep_ind):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]                              
        indep_samp_prob = np.array([m/M] * M)
        np.random.seed(seed0+rep_ind*12+M+get_seed(simu_type, ml_name, lossfunc))
        res = LOCOAdaMP(X, Y, X1, Y1, n, m, Kb, ml_model, indep_samp_prob, alpha, lossfunc, indep_delta, selected_features = [], bonf=False)
        f = open(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl','wb')
        pickle.dump(res,f)
        f.close()    
    Parallel(n_jobs=n_jobs)(delayed(get_predci_LOCOAdaMP)(rep_ind) for rep_ind in range(rep))

### Result for LOCO-AdaMP
def res_pred_ci_LOCOAdaMP(M, simu_type, ml_name, rep, alpha, target_inds, n_features, lossfunc, output_path = '.'):
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_1.pkl')

    adap_pred_err_all = np.zeros((rep, len(result)))
    adap_train_err_all = np.zeros((rep, len(result)))
    adap_target_all = np.zeros((rep, len(target_inds), len(result)))
    adap_ass = []

    for rep_ind in range(rep):
        result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl')
        for b in range(len(result)):
            zz = np.zeros(M)
            for j in range(M):
                zz[j] = result[b]['z'][j].mean()
            fea_ind = np.argmin(zz)
            prob_min = []
            if b>0:
                for b_ind in range(1,b+1):
                    prob_min.append(result[b_ind]['indep_samp_prob'][fea_ind])
                adap_ass.append(max(prob_min))

            adap_pred_err_all[rep_ind, b] = result[b]['err2'][0].mean()
            adap_train_err_all[rep_ind, b] = result[b]['resids_LOO'].mean()
            for s, tar in enumerate(target_inds):
                adap_target_all[rep_ind, s, b] = result[b]['target'][tar]

    adap_pred_err = adap_pred_err_all.mean(axis=0)
    adap_pred_err_std  = adap_pred_err_all.std(axis=0, ddof=1)
    adap_pred_err_yerr = norm.ppf(1-alpha/2) * adap_pred_err_std / np.sqrt(rep)

    adap_train_err = adap_train_err_all.mean(axis=0)
    adap_train_err_std  = adap_train_err_all.std(axis=0, ddof=1)
    adap_train_err_yerr = norm.ppf(1-alpha/2) * adap_train_err_std / np.sqrt(rep)

    adap_target = adap_target_all.mean(axis=0)
    adap_target_std  = adap_target_all.std(axis=0, ddof=1)
    adap_target_yerr = norm.ppf(1-alpha/2) * adap_target_std / np.sqrt(rep)

    pred_err = adap_pred_err[len(result)-1]  

    dfs_adap_ass = pd.DataFrame({'prob_min_max':adap_ass})
    file_path = f'{output_path}/fig_PredCI_AdaAss_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_adap_ass.to_pickle(file_path)

    dfs_adap_error = pd.DataFrame({'iteration':list(range(1,len(result)+1)), 'train_err':adap_train_err, 'train_err_err':adap_train_err_yerr, 'pred_err':adap_pred_err, 'pred_err_err':adap_pred_err_yerr})
    file_path = f'{output_path}/fig_PredCI_AdaError_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_adap_error.to_pickle(file_path)
    for s,tar in enumerate(target_inds):
        dfs_adap_target = pd.DataFrame({'iteration': list(range(1,len(result)+1)), 'target': adap_target[s], 'target_yerr': adap_target_yerr[s]})
        file_path = f'{output_path}/fig_PredCI_AdaTarget{tar}_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
        dfs_adap_target.to_pickle(file_path)
    
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_1.pkl')
    adap_ci_center = [0]*n_features
    adap_ci_err = [0]*n_features
    indep_samp_prob = [0]*n_features
    dfs_adap_samp_prob = {b: pd.DataFrame() for b in range(len(result))}
    dfs_adap_ci = {b: pd.DataFrame() for b in range(len(result))}
    for b in range(len(result)):
        for i in range(n_features):
            adap_ci_center[i]=0.5*(result[b]['inf'][i][2]+result[b]['inf'][i][3])
            adap_ci_err[i]=0.5*(result[b]['inf'][i][3]-result[b]['inf'][i][2])
            indep_samp_prob[i] = result[b]['indep_samp_prob'][i]
        adap_ci = pd.DataFrame({'feature':list(range(1,n_features+1)), 'ci_center':adap_ci_center, 'ci_err':adap_ci_err})
        if b == len(result)-1:
            dfs_ci= adap_ci
        sampling_prob = pd.DataFrame({'feature':list(range(1,n_features+1)), 'indep_samp_prob':indep_samp_prob})
        dfs_adap_ci[b] = adap_ci
        file_path = f'{output_path}/fig_PredCI_AdaCI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_epoch{b}.pkl'
        dfs_adap_ci[b].to_pickle(file_path)
        dfs_adap_samp_prob[b] = sampling_prob
        file_path = f'{output_path}/fig_PredCI_AdaSampProb_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_epoch{b}.pkl'
        dfs_adap_samp_prob[b].to_pickle(file_path)
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Prediction for LOCO-MP
def pred_ci_LOCOMP(M, N, simu_type, ml_name, rep, alpha, n_jobs, lossfunc, simu_path = '.', output_path = '.', seed0=1):
    m,n,Kb,K_locomp = get_parameter(M,N)
    ml_model = get_model(ml_name)
    def get_predci_LOCOMP(rep_ind):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
        np.random.seed(seed0+rep_ind*78+M*123+31*get_seed(simu_type, ml_name, lossfunc))
        res = LOCOMP(X,Y,X1,Y1, n, m, K_locomp, ml_model, alpha, lossfunc, selected_features=[], bonf=False)
        f = open(f'{output_path}/res_PredCI_LOCO-MP_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl','wb')
        pickle.dump(res,f)
        f.close()
    Parallel(n_jobs=n_jobs)(delayed(get_predci_LOCOMP)(rep_ind) for rep_ind in range(rep))

### Result for LOCO-MP
def res_pred_ci_LOCOMP(M, simu_type, ml_name, rep, n_features, lossfunc, output_path = '.'): 
    pred_err = 0
    for rep_ind in range(rep):
        result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-MP_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl')      
        pred_err = pred_err + result['err2'][0].mean()/rep
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-MP_{simu_type}_{ml_name}_{lossfunc}_M{M}_1.pkl')
    ci_center = [0]*n_features
    ci_err = [0]*n_features
    for i in range(n_features):
        ci_center[i]=0.5*(result['inf'][i][2]+result['inf'][i][3])
        ci_err[i]=0.5*(result['inf'][i][3]-result['inf'][i][2])
    ci = pd.DataFrame({'feature':list(range(1,n_features+1)), 'ci_center':ci_center, 'ci_err':ci_err})
    dfs_ci = ci
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-MP_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Prediction for LOCO-Split
def pred_ci_LOCOSplit(M, simu_type, ml_name, rep, ratios, alpha, n_jobs, lossfunc, simu_path = '.', output_path = '.',seed0=1):
    ml_model = get_model_split(ml_name)
    def get_predci_LOCOSplit(rep_ind):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
        for ratio in ratios:
            np.random.seed(seed0+rep_ind*33+M*12+29*get_seed(simu_type, ml_name, lossfunc)+int(ratio*100))
            res = LOCOSplit(X, Y, X1, Y1, ml_model, ratio, alpha, lossfunc, selected_features=[], bonf=False)
            f = open(f'{output_path}/res_PredCI_LOCO-Split{ratio}_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl','wb')
            pickle.dump(res,f)
            f.close()
    Parallel(n_jobs=n_jobs)(delayed(get_predci_LOCOSplit)(rep_ind) for rep_ind in range(rep))


### Result for LOCO-Split
def res_pred_ci_LOCOSplit(M, simu_type, ml_name, rep, ratio, n_features, lossfunc, output_path = '.'):
    pred_err = 0
    for rep_ind in range(rep):
        result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-Split{ratio}_{simu_type}_{ml_name}_{lossfunc}_M{M}_{rep_ind}.pkl')      
        pred_err = pred_err + result['err2'][0].mean()/rep
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-Split{ratio}_{simu_type}_{ml_name}_{lossfunc}_M{M}_1.pkl')
    ci_center = [0]*n_features
    ci_err = [0]*n_features
    for i in range(n_features):
        ci_center[i]=0.5*(result['inf'][i][2]+result['inf'][i][3])
        ci_err[i]=0.5*(result['inf'][i][3]-result['inf'][i][2])
    ci = pd.DataFrame({'feature':list(range(1,n_features+1)), 'ci_center':ci_center, 'ci_err':ci_err})
    dfs_ci= ci
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-Split{ratio}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Comparison Figure 
def fig_pred_ci_LOCO(M, simu_type, ml_name, loco_meths, dfs_pred_error, dfs_ci, lossfunc, output_path = '.'):
    # Confidence Interval
    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd', "#8c564b"]
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
    for l, loco_meth in enumerate(loco_meths):            
        ci = dfs_ci[f'{loco_meth}']
        ax.errorbar(x=ci['feature']+l*0.135, y=ci['ci_center'], yerr=ci['ci_err'], color = color_list[l],linestyle="None", capsize=3, marker="_", label = f'{loco_meth}')
    ax.set_xticks(ci['feature'])
    ax.set_xlabel('Feature')
    ax.set_ylabel('90% Confidence Interval')
    ml_name1 = ml_rename(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.4, 0.5),loc='lower center')
    plt.tight_layout()
    plt.savefig(f'TargetCI_Comparison_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Prediction Error
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    FullData_pred = [dfs_pred_error.iloc[0,0]]
    LOCOAdaMP_pred= [dfs_pred_error.iloc[1,0]]
    LOCOMP_pred = [dfs_pred_error.iloc[2,0] ]
    LOCOSplit1_pred = [dfs_pred_error.iloc[3,0]]
    LOCOSplit2_pred = [dfs_pred_error.iloc[4,0]]
     
    barWidth = 0.1
    br1 = np.arange(1) 
    br2 = [x + barWidth for x in br1]  
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]

    ax.bar(br1, FullData_pred, width = barWidth, color= color_list[4], zorder=10, label ='Full Data') 
    ax.bar(br2, LOCOAdaMP_pred, width = barWidth, color= color_list[0], zorder=10, label ='LOCO-AdaMP')
    ax.bar(br3, LOCOMP_pred, width = barWidth, color= color_list[1], zorder=10, label ='LOCO-MP')
    ax.bar(br4, LOCOSplit1_pred, width = barWidth, color= color_list[2], zorder=10, label ='LOCO-Split0.5')
    ax.bar(br5, LOCOSplit2_pred, width = barWidth, color= color_list[3], zorder=10, label ='LOCO-Split0.75')
    ax.set_xticks([])
    ml_name1 = ml_rename(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_xlabel(ml_name1)
    ax.set_ylabel('Test Error')
    ax.set_title(f'{simu_type1} Model')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.4, 0.2),loc='lower center')
    plt.tight_layout()
    plt.savefig(f'TestError_Comparison_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

### Figure for LOCO-AdaMp
def fig_pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci, dfs_adap_ass, lossfunc):
    # Prediction Error
    fig, ax = plt.subplots(figsize=(3.7, 3))
    error = dfs_adap_error
    ax.plot(error['iteration'],error['pred_err'],linestyle='-', marker='o')
    ax.set_xticks(error['iteration'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test Error')
    ax.set_ylim([-0.05, 1.05])
    ml_name1 = ml_rename1(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'AdaMP_TestError_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Training Error
    fig, ax = plt.subplots(figsize=(3.7, 3))
    error = dfs_adap_error   
    ax.plot(error['iteration'],error['train_err'],linestyle='-', marker='o')
    ax.set_xticks(error['iteration'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Traning Error')
    ax.set_ylim([-0.05, 1.05])
    ml_name1 = ml_rename1(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'AdaMP_TrainError_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Assumption Check
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(dfs_adap_ass['prob_min_max'], rwidth=0.8)
    m,n,Kb,K_locomp = get_parameter(M,N)
    ax.axvline(x=m/M, color='red', linestyle='--', linewidth=1, label = "m/M")
    ax.set_xlabel('Probability')
    ax.set_xlim([-0.01, 0.31])
    ax.set_ylabel('Frequency')
    ml_name1 = ml_rename1(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.2, 0.5),loc='lower center') 
    plt.tight_layout()
    plt.savefig(f'AdaMP_Ass_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Target
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for tar in target_inds:
        target = dfs_adap_target[tar]   
        ax.plot(target['iteration'],target['target'],linestyle='-', marker='o', label = f'Feature {tar+1}')
    ax.set_xticks(target['iteration'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Target')
    #ax.set_yscale('log')
    ml_name1 = ml_rename1(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.4, 0.5),loc='lower center')  
    plt.tight_layout()
    plt.savefig(f'AdaMP_Target_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 3))
    for tar in target_inds:
        target = dfs_adap_target[tar]       
        ax.plot(target['iteration'],target['target'],linestyle='-', marker='o', label = f'Feature {tar+1}')
    ax.set_xticks(target['iteration'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Target')
    ax.set_yscale('symlog', linthresh=0.001)
    ax.set_ylim([-0.005, 0.5])
    ml_name1 = ml_rename1(ml_name)
    simu_type1 = simu_type_rename(simu_type)
    ax.set_title(f'{simu_type1} | {ml_name1}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.4, 0.5),loc='lower center')  
    plt.tight_layout()
    plt.savefig(f'AdaMP_LogTarget_{simu_type}_{ml_name}_{lossfunc}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
