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

### Run power functions
def inf_power(M, N, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, indep_delta, n_jobs, lossfunc, simu_path = '.', output_path = '.',seed0=1):
    power_LOCOAdaMP(M, N, simu_type, ml_name, betas, rep, beta_ind, alpha, indep_delta, n_jobs, lossfunc, simu_path = simu_path, output_path = output_path,seed0=seed0)
    power_LOCOMP(M, N, simu_type, ml_name, betas, rep, beta_ind, alpha, n_jobs, lossfunc, simu_path = simu_path, output_path = output_path,seed0=seed0)
    power_LOCOSplit(M, N, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, n_jobs, lossfunc, simu_path = simu_path, output_path = output_path,seed0=seed0)

def res_power(M, loco_meths, simu_type, ml_name, betas, rep, alpha, beta_ind, pvalue_thre, lossfunc, output_path = '.'):
    for loco_meth in loco_meths:
        res_power_LOCO(M, loco_meth, simu_type, ml_name, betas, rep, alpha, beta_ind, pvalue_thre, lossfunc, output_path = output_path)

def fig_power(M, loco_meths, simu_type, ml_name, lossfunc, output_path = '.'):
    dfs_power = {loco_meth: pd.DataFrame() for loco_meth in loco_meths}
    dfs_target = {loco_meth: pd.DataFrame() for loco_meth in loco_meths}
    for loco_meth in loco_meths:
        dfs_power[f'{loco_meth}'] = pd.read_pickle(f'{output_path}/fig_Power_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
        dfs_target[f'{loco_meth}'] = pd.read_pickle(f'{output_path}/fig_PowerTarget_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl')
    fig_power_LOCO(M, simu_type, ml_name, loco_meths, dfs_power, dfs_target, lossfunc)


### Power for LOCO-AdaMP
def power_LOCOAdaMP(M, N, simu_type, ml_name, betas, rep, beta_ind, alpha, indep_delta, n_jobs, lossfunc, simu_path = '.', output_path = '.',seed0=1):
    m,n,Kb,K_locomp = get_parameter(M,N)
    ml_model = get_model(ml_name)
    
    def get_power_LOCOAdaMP(beta, rep_ind):          
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_Beta{beta[beta_ind]}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]              
        indep_samp_prob = np.array([m/M] * M)       
        np.random.seed(seed0+rep_ind*65+M*43+77*get_seed(simu_type, ml_name, lossfunc)+int(145*beta[beta_ind]))
        res = LOCOAdaMP(X, Y, X1, Y1, n, m, Kb, ml_model, indep_samp_prob, alpha, lossfunc, indep_delta, selected_features = [], bonf=False)
        f = open(f'{output_path}/res_Power_LOCO-AdaMP_{simu_type}_{ml_name}_{lossfunc}_M{M}_Beta{beta[beta_ind]}_rep{rep_ind}.pkl','wb')
        pickle.dump(res[4],f)
        f.close()
    Parallel(n_jobs=n_jobs)(delayed(get_power_LOCOAdaMP)(beta, rep_ind) for beta in betas for rep_ind in range(rep))


### Power for LOCO-MP
def power_LOCOMP(M, N, simu_type, ml_name, betas, rep, beta_ind, alpha, n_jobs, lossfunc, simu_path = '.', output_path = '.',seed0=1):
    m,n,Kb,K_locomp = get_parameter(M,N)
    K_locomp = sum(Kb)
    ml_model = get_model(ml_name)

    def get_power_LOCOMP(beta, rep_ind):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_Beta{beta[beta_ind]}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
        np.random.seed(seed0+rep_ind*27+M*11+24*get_seed(simu_type, ml_name, lossfunc)+int(1720*beta[beta_ind]))
        res = LOCOMP(X, Y, X1, Y1, n, m, K_locomp, ml_model, alpha, lossfunc, selected_features=[], bonf=False)
        ress = res
        f = open(f'{output_path}/res_Power_LOCO-MP_{simu_type}_{ml_name}_{lossfunc}_M{M}_Beta{beta[beta_ind]}_rep{rep_ind}.pkl','wb')
        pickle.dump(ress,f)
        f.close()

    Parallel(n_jobs=n_jobs)(delayed(get_power_LOCOMP)(beta, rep_ind) for beta in betas for rep_ind in range(rep))


### Power for LOCO-Split
def power_LOCOSplit(M, N, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, n_jobs, lossfunc, simu_path = '.', output_path = '.',seed0=1):
    ml_model = get_model_split(ml_name)

    def get_power_LOCOSplit(ratio, beta, rep_ind):
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_Beta{beta[beta_ind]}_M{M}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]
        np.random.seed(seed0+rep_ind*19+M*321+93*get_seed(simu_type, ml_name, lossfunc)+int(ratio*100+589*beta[beta_ind]))
        res = LOCOSplit(X, Y, X1, Y1, ml_model, ratio, alpha, lossfunc, selected_features=[], bonf=False)
        ress = res
        f = open(f'{output_path}/res_Power_LOCO-Split{ratio}_{simu_type}_{ml_name}_{lossfunc}_M{M}_Beta{beta[beta_ind]}_rep{rep_ind}.pkl','wb')
        pickle.dump(ress,f)
        f.close()

    Parallel(n_jobs=n_jobs)(delayed(get_power_LOCOSplit)(ratio, beta, rep_ind) for ratio in ratios for beta in betas for rep_ind in range(rep))


### Result for LOCO methods
def res_power_LOCO(M, loco_meth, simu_type, ml_name, betas, rep, alpha, beta_ind, pvalue_thre, lossfunc, output_path = '.'):
    beta_1 = [beta[beta_ind] for beta in betas]       
    power, err = [0]*len(betas), [0]*len(betas)
    target, z = [0]*rep, [0]*rep
    for k, beta in enumerate(betas):
        count = 0
        for rep_ind in range(rep):
            res = pd.read_pickle(f'{output_path}/res_Power_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}_Beta{beta[beta_ind]}_rep{rep_ind}.pkl')
                   
            pvalue = res['inf'][beta_ind][0]                   
            if (pvalue<pvalue_thre):
                count += 1
            if betas[k][beta_ind] == 0:
                target[rep_ind] = res['target'][beta_ind]
                z[rep_ind] = res['z'][beta_ind].mean()
        power[k] = count/rep
        err[k] = norm.ppf(1-alpha/2) * np.sqrt(power[k]*(1-power[k])/rep)
    dfs_power = pd.DataFrame({'beta': beta_1, 'power':power, 'err':err})
    file_path = f'{output_path}/fig_Power_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_power.to_pickle(file_path)
    dfs_target = pd.DataFrame({'target': target, 'z':z})
    file_path = f'{output_path}/fig_PowerTarget_{loco_meth}_{simu_type}_{ml_name}_{lossfunc}_M{M}.pkl'
    dfs_target.to_pickle(file_path)


### Figures
def fig_power_LOCO(M, simu_type, ml_name, loco_meths, dfs_power, dfs_target, lossfunc):
    # Power Comparison
    # Power
    def get_fig_power(ci = True):
        color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd', "#8c564b"]
        fig, ax = plt.subplots(figsize=(6, 2.8))
        for l, loco_meth in enumerate(loco_meths):  
            zo = 1
            if loco_meth == "LOCO-AdaMP":
                zo = 10                  
            power = dfs_power[f'{loco_meth}']
            if ci:
                ax.errorbar(x=power['beta'],y=power['power'],yerr=power['err'],color = color_list[l],linestyle='-', capsize=3, marker='o', label = f'{loco_meth}', zorder=zo)
            else:
                ax.plot(power['beta'],power['power'],color = color_list[l],linestyle='-', marker='o', label = f'{loco_meth}', zorder=zo)
        ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='0.1 Line')
        ax.set_xlabel(r'$\beta_5$')
        ax.set_xticks(power['beta'])
        ax.set_ylim([-0.03, 1.03])
        ax.set_ylabel('Power')
        ml_name1 = ml_rename(ml_name)
        simu_type1 = simu_type_rename(simu_type)
        ax.set_title(f'{simu_type1} | {ml_name1}')
        ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
        ax.legend(bbox_to_anchor=(1.35, 0.3),loc='lower center')
        plt.tight_layout()
        if ci:
            plt.savefig(f'PowerCI_Comparison_{simu_type}_{ml_name}_{lossfunc}_M{M}.pdf', bbox_inches="tight")
        else:
            plt.savefig(f'Power_Comparison_{simu_type}_{ml_name}_{lossfunc}_M{M}.pdf', bbox_inches="tight")
        plt.show()
        plt.close()

    get_fig_power(ci = True)
    get_fig_power(ci = False)

