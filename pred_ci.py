import numpy as np
import pandas as pd
import pickle
from loco_methods import *
from ml_models import *
from simulations import *
from utils import *
from scipy.stats import *
import matplotlib.pyplot as plt

### Run prediction functions
def inf_pred_ci(M, N, simu_type, ml_name, ratios, alpha, indep_delta, simu_path = '.', output_path = '.'):
    pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, alpha, indep_delta, simu_path = simu_path, output_path = output_path) 
    pred_ci_LOCOMP(M, N, simu_type, ml_name, alpha, simu_path = simu_path, output_path = output_path)
    pred_ci_LOCOSplit(M, simu_type, ml_name, ratios, alpha, simu_path = simu_path, output_path = output_path)

def res_pred_ci(M, loco_meths, simu_type, ml_name, ratios, target_inds, n_features, simu_path = '.', output_path = '.'):
    pred_err = []
    data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}.pkl')
    X,Y,X1,Y1 = data[0], data[1], data[2], data[3] 
    ml_model = get_model_split(ml_name)
    pred_err.append(((ml_model(X,Y,X1) - Y1)**2).mean())
    pred_err.append(res_pred_ci_LOCOAdaMP(M, simu_type, ml_name, target_inds, n_features, output_path = output_path))

    pred_err.append(res_pred_ci_LOCOMP(M, simu_type, ml_name, n_features, output_path = output_path))
    for ratio in ratios:
        pred_err.append(res_pred_ci_LOCOSplit(M, simu_type, ml_name, ratio, n_features, output_path = output_path))

    dfs_pred_error = pd.DataFrame({'loco_meth':['Full-Data']+loco_meths, 'pred_err': pred_err})
    dfs_pred_error = dfs_pred_error.set_index('loco_meth')
    file_path = f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_pred_error.to_pickle(file_path)

def fig_pred_ci(M, loco_meths, simu_type, ml_name, indep_delta, target_inds, output_path = '.', figure_path = '.'):
    dfs_ci = {loco_meth: pd.DataFrame() for loco_meth in loco_meths}
    dfs_pred_error = pd.read_pickle(f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_M{M}.pkl')
    dfs_adap_error = result = pd.read_pickle(f'{output_path}/fig_PredCI_AdaError_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
    dfs_adap_samp_prob = {b: pd.DataFrame() for b in range(len(result))}
    dfs_adap_ci = {b: pd.DataFrame() for b in range(len(result))}
    dfs_adap_target = {tar: pd.DataFrame() for tar in target_inds}
    for epo in range(len(result)):
        dfs_adap_samp_prob[epo] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaSampProb_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{epo}.pkl')
        dfs_adap_ci[epo]= pd.read_pickle(f'{output_path}/fig_PredCI_AdaCI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{epo}.pkl')
    for loco_meth in loco_meths:
        dfs_ci[f'{loco_meth}'] = pd.read_pickle(f'{output_path}/fig_PredCI_CI_{loco_meth}_{simu_type}_{ml_name}_M{M}.pkl')
    for tar in target_inds:
        dfs_adap_target[tar] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaTarget{tar}_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
    fig_pred_ci_LOCO(M, simu_type, ml_name, loco_meths, dfs_pred_error, dfs_ci, output_path = output_path, figure_path = figure_path)
    fig_pred_ci_LOCOAdaMP(M, simu_type, ml_name, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci,figure_path = figure_path)

### Prediction for LOCO-AdaMP
def pred_ci_LOCOAdaMP(M, N, simu_type, ml_name, alpha, indep_delta, simu_path = '.', output_path = '.'):
    m,n = get_mn(M,N)
    Kb = get_Kb(M, N, m, n)
    data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}.pkl')
    X,Y,X1,Y1 = data[0], data[1], data[2], data[3]                              
    indep_samp_prob = np.array([1/M] * M)
    ml_model = get_model(ml_name)
    ress={}
    for epo in range(len(Kb)):
        B = Kb[epo]
        res = LOCOAdaMP(X,Y,X1,Y1, n, m, B, ml_model, indep_samp_prob, alpha=alpha, selected_features=[])
        ress[epo] = res            
        f = open(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl','wb')
        pickle.dump(ress,f)
        f.close()    
        indep_samp_prob = indep_sampling_probability(res['z'], m, indep_delta)

### Result for LOCO-AdaMP
def res_pred_ci_LOCOAdaMP(M, simu_type, ml_name, target_inds, n_features, output_path = '.'):
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')      
    adap_pred_err = [0]*len(result)
    adap_train_err = [0]*len(result)
    adap_target = [[0]*len(result) for _ in target_inds]
    for b in range(len(result)):
        adap_pred_err[b]=result[b]['err2_sq'][0].mean()
        adap_train_err[b] = result[b]['resids_LOO_sq'].mean()
        for s,tar in enumerate(target_inds):
            adap_target[s][b] = result[b]['target'][tar]
    pred_err = adap_pred_err[len(result)-1]          
    dfs_adap_error = pd.DataFrame({'epoch':list(range(1,len(result)+1)), 'train_err':adap_train_err, 'pred_err':adap_pred_err})
    file_path = f'{output_path}/fig_PredCI_AdaError_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_adap_error.to_pickle(file_path)
    for s,tar in enumerate(target_inds):
        dfs_adap_target = pd.DataFrame({'epoch': list(range(1,len(result)+1)), 'target': adap_target[s]})
        file_path = f'{output_path}/fig_PredCI_AdaTarget{tar}_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl'
        dfs_adap_target.to_pickle(file_path)
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
        file_path = f'{output_path}/fig_PredCI_AdaCI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{b}.pkl'
        dfs_adap_ci[b].to_pickle(file_path)
        dfs_adap_samp_prob[b] = sampling_prob
        file_path = f'{output_path}/fig_PredCI_AdaSampProb_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{b}.pkl'
        dfs_adap_samp_prob[b].to_pickle(file_path)
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Prediction for LOCO-MP
def pred_ci_LOCOMP(M, N, simu_type, ml_name, alpha, simu_path = '.', output_path = '.'):
    m,n = get_mn(M,N)
    Kb = get_Kb(M, N, m, n)
    K_locomp = sum(Kb)
    data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}.pkl')
    X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
    ml_model = get_model(ml_name)
    res = LOCOMP(X,Y,X1,Y1, n, m, K_locomp, ml_model, alpha=alpha, selected_features=[], bonf=False)
    f = open(f'{output_path}/res_PredCI_LOCO-MP_{simu_type}_{ml_name}_M{M}.pkl','wb')
    pickle.dump(res,f)
    f.close()

### Result for LOCO-MP
def res_pred_ci_LOCOMP(M, simu_type, ml_name, n_features, output_path = '.'):    
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-MP_{simu_type}_{ml_name}_M{M}.pkl')      
    pred_err = result['err2_sq'][0].mean()      
    ci_center = [0]*n_features
    ci_err = [0]*n_features
    for i in range(n_features):
        ci_center[i]=0.5*(result['inf'][i][2]+result['inf'][i][3])
        ci_err[i]=0.5*(result['inf'][i][3]-result['inf'][i][2])
    ci = pd.DataFrame({'feature':list(range(1,n_features+1)), 'ci_center':ci_center, 'ci_err':ci_err})
    dfs_ci = ci
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-MP_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Prediction for LOCO-Split
def pred_ci_LOCOSplit(M, simu_type, ml_name, ratios, alpha, simu_path = '.', output_path = '.'):
    data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}.pkl')
    X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
    ml_model = get_model_split(ml_name)
    for ratio in ratios:            
        res = LOCOSplit(X, Y, X1, Y1, ml_model, ratio, alpha=alpha, selected_features=[], bonf=False)
        f = open(f'{output_path}/res_PredCI_LOCO-Split{ratio}_{simu_type}_{ml_name}_M{M}.pkl','wb')
        pickle.dump(res,f)
        f.close()

### Result for LOCO-Split
def res_pred_ci_LOCOSplit(M, simu_type, ml_name, ratio, n_features, output_path = '.'):
    result = pd.read_pickle(f'{output_path}/res_PredCI_LOCO-Split{ratio}_{simu_type}_{ml_name}_M{M}.pkl')      
    pred_err = result['err2_sq'][0].mean()
    ci_center = [0]*n_features
    ci_err = [0]*n_features
    for i in range(n_features):
        ci_center[i]=0.5*(result['inf'][i][2]+result['inf'][i][3])
        ci_err[i]=0.5*(result['inf'][i][3]-result['inf'][i][2])
    ci = pd.DataFrame({'feature':list(range(1,n_features+1)), 'ci_center':ci_center, 'ci_err':ci_err})
    dfs_ci= ci
    file_path = f'{output_path}/fig_PredCI_CI_LOCO-Split{ratio}_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_ci.to_pickle(file_path)
    return pred_err

### Comparison Figure 
def fig_pred_ci_LOCO(M, simu_type, ml_name, loco_meths, dfs_pred_error, dfs_ci, output_path = '.', figure_path = '.'):
    # Confidence Interval
    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd']#red, green, orange, blue, purple
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
    for l, loco_meth in enumerate(loco_meths):            
        ci = dfs_ci[f'{loco_meth}']
        ax.errorbar(x=ci['feature']+l*0.1, y=ci['ci_center'], yerr=ci['ci_err'], color = color_list[l],linestyle="None", capsize=3, marker="_", label = f'{loco_meth}')
    ax.set_xticks(ci['feature'])
    ax.set_xlabel('Feature')
    ax.set_ylabel('90% Confidence Interval')
    ax.set_title(f'{simu_type} | {ml_name}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.6, 0.5),loc='right')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/TargetCI_Comparison_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Prediction Error
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
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

    ax.bar(br1, FullData_pred, width = barWidth, color= color_list[4], zorder=10, label ='Baseline (Full Data)') 
    ax.bar(br2, LOCOAdaMP_pred, width = barWidth, color= color_list[0], zorder=10, label ='AdaMP')
    ax.bar(br3, LOCOMP_pred, width = barWidth, color= color_list[1], zorder=10, label ='MP')
    ax.bar(br4, LOCOSplit1_pred, width = barWidth, color= color_list[2], zorder=10, label ='Baseline (50% Data)')
    ax.bar(br5, LOCOSplit2_pred, width = barWidth, color= color_list[3], zorder=10, label ='Baseline (75% Data)')
    ax.set_xticks([])
    ax.set_xlabel(ml_name)
    ax.set_ylabel('Prediction Error')
    ax.set_title(f'{simu_type} Model')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.65, 0.5),loc='right')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/PredError_Comparison_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

### Figure for LOCO-AdaMp
def fig_pred_ci_LOCOAdaMP(M, simu_type, ml_name, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci, figure_path = '.'):
    # Prediction Error
    fig, ax = plt.subplots(figsize=(3.7, 3))
    error = dfs_adap_error
    ax.plot(error['epoch'],error['pred_err'],linestyle='-', marker='o')
    ax.set_xticks(error['epoch'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Error')
    ax.set_ylim([-0.03, 1.03])
    ax.set_title(f'{simu_type} | {ml_name}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_PredError_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Training Error
    fig, ax = plt.subplots(figsize=(3.7, 3))
    error = dfs_adap_error   
    ax.plot(error['epoch'],error['train_err'],linestyle='-', marker='o')
    ax.set_xticks(error['epoch'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Traning Error')
    ax.set_ylim([-0.03, 1.03])
    ax.set_title(f'{simu_type} | {ml_name}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_TrainError_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Sampling Probability
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(y=indep_delta, color='red', linestyle='--', linewidth=1, label='Delta Line (0.9)')
    for b in range(len(error['epoch'])):            
        sampling_prob = dfs_adap_samp_prob[b]
        ax.plot(sampling_prob['feature'],sampling_prob['indep_samp_prob'],linestyle='-', marker='o', label = f'Epoch {b+1}')
    ax.set_xticks(sampling_prob['feature'])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Sampling Probability')
    ax.set_title(f'{simu_type} | {ml_name}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.65, 0.5),loc='right')  
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_SampProb_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Target
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for tar in target_inds:
        target = dfs_adap_target[tar]       
        ax.plot(target['epoch'],target['target'],linestyle='-', marker='o', label = f'Target{tar+1}')
    ax.set_xticks(target['epoch'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Target')
    ax.set_title(f'{simu_type} | {ml_name}')
    ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    ax.legend(bbox_to_anchor=(1.4, 0.5),loc='right')  
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_Target_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Confidence Interval        
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    ax = ax.flatten()
    for b in range(len(error['epoch'])):
        ci = dfs_adap_ci[b]
        ax[b].errorbar(x=ci['feature'], y=ci['ci_center'], yerr=ci['ci_err'], linestyle="None", capsize=3, marker="_")
        ax[b].axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
        ax[b].set_xticks(ci['feature'])
        ax[b].set_xlabel('Feature')
        ax[b].set_ylabel('90% Confidence Interval')
        ax[b].set_title(f'{simu_type} | {ml_name} | Epoch {b+1}')
        ax[b].grid(True, zorder=0, alpha=0.5, linestyle='--')           
    fig.delaxes(ax[5])
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaCI/AdaMP_TargetCI_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
