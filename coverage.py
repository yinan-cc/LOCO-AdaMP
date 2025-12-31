import numpy as np
import pandas as pd
import pickle
from loco_methods import *
from ml_models import *
from utils import *
from simulations import *
from scipy.stats import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

### Run coverage functions
def inf_coverage(M, Ns, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, simu_path = '.', output_path = '.'):
    cover_LOCOAdaMP(M, Ns, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, simu_path = simu_path, output_path = output_path)

def res_cover(M, Ns, simu_type, ml_name, rep, alpha, signal, noise, output_path = '.'):
    res_cover_LOCOAdaMP(M, Ns, simu_type, ml_name, rep, alpha, signal, noise, output_path = output_path)

def fig_cover(M, simu_type, ml_name, output_path = '.', figure_path = '.'):
    dfs_coverage = pd.read_pickle(f'{output_path}/fig_Coverage_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
    fig_cover_LOCOAdaMP(M, simu_type, ml_name, dfs_coverage, figure_path = figure_path)


### Coverage for LOCO-AdaMP
def cover_LOCOAdaMP(M, Ns, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, simu_path = '.', output_path = '.'):
    ml_model = get_model(ml_name)

    def coverage_LOCOAdaMP(N, rep_ind):
        m,n = get_mn(M,N)
        Kb = get_Kb(M,N,m,n)
        data =  pd.read_pickle(f'{simu_path}/sim_{simu_type}_M{M}_N{N}_{rep_ind}.pkl')
        X,Y,X1,Y1 = data[0], data[1], data[2], data[3]              
        indep_samp_prob = np.array([1/M] * M)
        ml_name = get_model_name(ml_model)
        for epo in range(len(Kb)):
            B = Kb[epo]
            res = LOCOAdaMP(X, Y, X1, Y1, n, m, B, ml_model, indep_samp_prob, alpha=alpha, selected_features=[])
            if epo == len(Kb)-1:
                f = open(f'{output_path}/res_Coverage_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_N{N}_rep{rep_ind}.pkl','wb')
                pickle.dump(res,f)
                f.close()                    
            indep_samp_prob = indep_sampling_probability(res['z'], m, indep_delta)

    Parallel(n_jobs=n_jobs)(delayed(coverage_LOCOAdaMP)(N, rep_ind) for N in Ns for rep_ind in range(rep))

### Result for LOCO-AdaMP
def res_cover_LOCOAdaMP(M, Ns, simu_type, ml_name, rep, alpha, signal, noise, output_path = '.'):
    coverage_s, coverage_n, err_s, err_n = [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns)
    width_s, width_n, target_s, target_n, var_s, var_n = [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns)
    coverage_adj_s, coverage_adj_n, err_adj_s, err_adj_n, width_adj_s, width_adj_n = [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns), [0]*len(Ns)
    for k, N in enumerate(Ns):
        coverage_s1, coverage_n1, width_s1, width_n1, target_s1, target_n1, var_s1, var_n1 = [0]*rep, [0]*rep, [0]*rep, [0]*rep, [0]*rep, [0]*rep, [0]*rep, [0]*rep
        coverage_adj_s1, coverage_adj_n1, width_adj_s1, width_adj_n1 = [0]*rep, [0]*rep, [0]*rep, [0]*rep
        for rep_ind in range(rep):
            result= pd.read_pickle(f'{output_path}/res_Coverage_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_N{N}_rep{rep_ind}.pkl')
            target_s1[rep_ind] = result['target'][signal]
            var_s1[rep_ind] = result['variance'][signal]
            width_s1[rep_ind] = (result['inf'][signal][3]-result['inf'][signal][2])
            if ((target_s1[rep_ind]<result['inf'][signal][3]) and (target_s1[rep_ind]>result['inf'][signal][2])):
                coverage_s1[rep_ind] = 1
                    
            target_n1[rep_ind] = result['target'][noise]
            var_n1[rep_ind] = result['variance'][noise]
            width_n1[rep_ind] = (result['inf'][noise][3]-result['inf'][noise][2])
            if ((target_n1[rep_ind]<result['inf'][noise][3]) and (target_n1[rep_ind]>result['inf'][noise][2])):
                coverage_n1[rep_ind] = 1
                
            ### Adjusted version
            width_adj_s1[rep_ind] = (result['adj_ci'][signal][1]-result['adj_ci'][signal][0])
            if ((target_s1[rep_ind]<result['adj_ci'][signal][1]) and (target_s1[rep_ind]>result['adj_ci'][signal][0])):
                coverage_adj_s1[rep_ind] = 1
                    
            width_adj_n1[rep_ind] = (result['adj_ci'][noise][1]-result['adj_ci'][noise][0])
            if ((target_n1[rep_ind]<result['adj_ci'][noise][1]) and (target_n1[rep_ind]>result['adj_ci'][noise][0])):
                coverage_adj_n1[rep_ind] = 1
 
        coverage_s[k] = sum(coverage_s1)/len(coverage_s1)
        coverage_n[k] = sum(coverage_n1)/len(coverage_n1)
        width_s[k] = sum(width_s1)/len(width_s1)
        width_n[k] = sum(width_n1)/len(width_n1)
        target_s[k] = sum(target_s1)/len(target_s1)
        target_n[k] = sum(target_n1)/len(target_n1)
        var_s[k] = sum(var_s1)/len(var_s1)
        var_n[k] = sum(var_n1)/len(var_n1)
        err_s[k] = norm.ppf(1-alpha/2) * np.sqrt(coverage_s[k]*(1-coverage_s[k])/rep)
        err_n[k] = norm.ppf(1-alpha/2) * np.sqrt(coverage_n[k]*(1-coverage_n[k])/rep)

        coverage_adj_s[k] = sum(coverage_adj_s1)/len(coverage_adj_s1)
        coverage_adj_n[k] = sum(coverage_adj_n1)/len(coverage_adj_n1)
        width_adj_s[k] = sum(width_adj_s1)/len(width_adj_s1)
        width_adj_n[k] = sum(width_adj_n1)/len(width_adj_n1)
        err_adj_s[k] = norm.ppf(1-alpha/2) * np.sqrt(coverage_adj_s[k]*(1-coverage_adj_s[k])/rep)
        err_adj_n[k] = norm.ppf(1-alpha/2) * np.sqrt(coverage_adj_n[k]*(1-coverage_adj_n[k])/rep)
        
    dfs_coverage= pd.DataFrame({'N': Ns, 'coverage_s':coverage_s, 'coverage_n':coverage_n,
                                'width_s':width_s, 'width_n':width_n, 'target_s': target_s, 'target_n': target_n,
                                'var_s':var_s, 'var_n':var_n, 'err_s':err_s, 'err_n':err_n, 
                                'coverage_adj_s':coverage_adj_s, 'coverage_adj_n':coverage_adj_n,
                                'width_adj_s':width_adj_s, 'width_adj_n':width_adj_n,
                                'err_adj_s':err_adj_s, 'err_adj_n':err_adj_n})
    file_path = f'{output_path}/fig_Coverage_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl'
    dfs_coverage.to_pickle(file_path)


### Figure for Coverage
def fig_cover_LOCOAdaMP(M, simu_type, ml_name, dfs_coverage, figure_path = '.'):
    # Coverage
    def fig_coverage(coverage_s, coverage_n, err_s, err_n, Coverage):
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='0.9 Line')       
        coverage = dfs_coverage
        ax.errorbar(x=coverage['N'], y=coverage[f'{coverage_s}'], yerr=coverage[f'{err_s}'], linestyle="-", capsize=3, marker="o", label = "Signal")
        ax.errorbar(x=coverage['N'], y=coverage[f'{coverage_n}'], yerr=coverage[f'{err_n}'], linestyle="-", capsize=3, marker="o", label = "Noise")
        ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
        ax.set_xlabel('N')
        ax.set_xticks(coverage['N'])
        ax.set_ylim([-0.03, 1.03])
        ax.set_ylabel('Coverage')
        ax.set_title(f'{simu_type} | {ml_name}')
        ax.legend(bbox_to_anchor=(1.42, 0.5),loc='right')  
        plt.tight_layout()
        plt.savefig(f'{figure_path}/{Coverage}_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    fig_coverage('coverage_s', 'coverage_n', 'err_s', 'err_n','Coverage')
    fig_coverage('coverage_adj_s', 'coverage_adj_n', 'err_adj_s', 'err_adj_n','Coverage_Adj')

    # Interval Width, Target, Variance
    def fig_cover_item(item_name, feature_name, ylabel_name, save_name):
        fig, ax = plt.subplots(figsize=(5.5, 3))
        coverage = dfs_coverage
        ax.plot(coverage['N'],coverage[f'{item_name}'],linestyle='-', marker='o', label = f'{feature_name}')
        ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
        ax.set_xlabel('N')
        ax.set_xticks(coverage['N'])
        ax.set_ylabel(f'{ylabel_name}')
        ax.set_title(f'{simu_type} | {ml_name}')
        ax.legend(bbox_to_anchor=(1.42, 0.5),loc='right')  
        plt.tight_layout()
        plt.savefig(f'{figure_path}/Coverage_{save_name}_{simu_type}_{ml_name}_M{M}.png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    fig_cover_item('width_s', 'Signal', 'Interval Width', 'Width_Signal')
    fig_cover_item('width_adj_s', 'Signal', 'Interval Width', 'Width_Signal_Adj')
    fig_cover_item('width_n', 'Noise', 'Interval Width', 'Width_Noise')
    fig_cover_item('width_adj_n', 'Noise', 'Interval Width', 'Width_Noise_Adj')
    fig_cover_item('target_s', 'Signal', 'Target', 'Target_Signal')
    fig_cover_item('target_n', 'Noise', 'Target', 'Target_Noise')
    fig_cover_item('var_s', 'Signal', 'Variance', 'Variance_Signal')
    fig_cover_item('var_n', 'Noise', 'Variance', 'Variance_Noise')
