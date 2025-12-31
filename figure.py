import numpy as np
import pandas as pd
from loco_methods import *
from ml_models import *
from simulations import *
from utils import *
from scipy.stats import *
import matplotlib.pyplot as plt

### PredCI
def fig_pred_ci_all(M, loco_meths, simu_types, ml_names, indep_delta, target_inds, output_path = '.', figure_path = '.'):
    dfs_pred_error = {loco_meth: {simu_type: {ml_name: 0 for ml_name in ml_names} for simu_type in simu_types}for loco_meth in ['Full-Data']+loco_meths}
    dfs_ci = {loco_meth: {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names}for simu_type in simu_types}for loco_meth in loco_meths}
    dfs_adap_error = {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types}
    dfs_adap_samp_prob= {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types}
    dfs_adap_ci= {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types}
    dfs_adap_target = {simu_type: {ml_name: {tar: pd.DataFrame() for tar in target_inds} for ml_name in ml_names} for simu_type in simu_types}
    for simu_type in simu_types:
        for ml_name in ml_names:
            dfs_adap_error[f'{simu_type}'][f'{ml_name}'] = result = pd.read_pickle(f'{output_path}/fig_PredCI_AdaError_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
            dfs_adap_samp_prob[f'{simu_type}'][f'{ml_name}'] = {b: pd.DataFrame() for b in range(len(result))}
            dfs_adap_ci[f'{simu_type}'][f'{ml_name}'] = {b: pd.DataFrame() for b in range(len(result))}
            for epo in range(len(result)):
                dfs_adap_samp_prob[f'{simu_type}'][f'{ml_name}'][epo] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaSampProb_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{epo}.pkl')
                dfs_adap_ci[f'{simu_type}'][f'{ml_name}'][epo]= pd.read_pickle(f'{output_path}/fig_PredCI_AdaCI_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}_epoch{epo}.pkl')

            dfs_pred_error['Full-Data'][f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_M{M}.pkl').iloc[0, 0]
            for i,loco_meth in enumerate(loco_meths):
                dfs_pred_error[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_PredCI_Error_{simu_type}_{ml_name}_M{M}.pkl').iloc[i+1, 0]
                dfs_ci[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_PredCI_CI_{loco_meth}_{simu_type}_{ml_name}_M{M}.pkl')
            for tar in target_inds:
                dfs_adap_target[f'{simu_type}'][f'{ml_name}'][tar] = pd.read_pickle(f'{output_path}/fig_PredCI_AdaTarget{tar}_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
    
    fig_pred_ci_LOCO_all(M, simu_types, ml_names, loco_meths, dfs_pred_error, dfs_ci, output_path = output_path, figure_path = figure_path)
    fig_pred_ci_LOCOAdaMP_all(M, simu_types, ml_names, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci,figure_path = figure_path)

def fig_cover_all(M, simu_types, ml_names, output_path = '.', figure_path = '.'):
    dfs_coverage = {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types}
    for simu_type in simu_types:
        for ml_name in ml_names:
            dfs_coverage[f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_Coverage_LOCO-AdaMP_{simu_type}_{ml_name}_M{M}.pkl')
    
    fig_cover_LOCOAdaMP_all(M, simu_types, ml_names, dfs_coverage, figure_path = figure_path)

def fig_power_all(M, loco_meths, simu_types, ml_names, output_path = '.', figure_path = '.'):
    dfs_power = {loco_meth: {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types} for loco_meth in loco_meths}
    dfs_target = {loco_meth: {simu_type: {ml_name: pd.DataFrame() for ml_name in ml_names} for simu_type in simu_types} for loco_meth in loco_meths}
    for loco_meth in loco_meths:
        for simu_type in simu_types:
            for ml_name in ml_names:
                dfs_power[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_Power_{loco_meth}_{simu_type}_{ml_name}_M{M}.pkl')
                dfs_target[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}'] = pd.read_pickle(f'{output_path}/fig_PowerTarget_{loco_meth}_{simu_type}_{ml_name}_M{M}.pkl')
    fig_power_LOCO(M, simu_types, ml_names, loco_meths, dfs_power, dfs_target, figure_path = figure_path)

### PredCI Comparison
def fig_pred_ci_LOCO_all(M, simu_types, ml_names, loco_meths, dfs_pred_error, dfs_ci, output_path = '.', figure_path = '.'):
    # Confidence Interval
    color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd']#red, green, orange, blue, purple
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for i,simu_type in enumerate(simu_types):   
        for j,ml_name in enumerate(ml_names):
            ax = axes[i,j]
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
            for l, loco_meth in enumerate(loco_meths):            
                ci = dfs_ci[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}']
                ax.errorbar(x=ci['feature']+l*0.1, y=ci['ci_center'], yerr=ci['ci_err'], color = color_list[l],linestyle="None", capsize=3, marker="_", label = f'{loco_meth}')
            ax.set_xticks(ci['feature'])
            ax.set_xlabel('Feature')
            ax.set_ylabel('90% Confidence Interval')
            ax.set_title(f'{simu_type} | {ml_name}')
            ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    handles, labels = axes[1,2].get_legend_handles_labels()
    axes[1,2].legend(handles, labels,bbox_to_anchor=(1.65, 0.5),loc='right')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/TargetCI_Comparison_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Prediction Error
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax = ax.flatten()
    for i,simu_type in enumerate(simu_types):
        FullData_pred = []
        LOCOAdaMP_pred= []
        LOCOMP_pred = []   
        LOCOSplit1_pred = []
        LOCOSplit2_pred = []
        for ml_name in ml_names:
            FullData_pred.append(dfs_pred_error[f'Full-Data'][f'{simu_type}'][f'{ml_name}'])
            LOCOAdaMP_pred.append(dfs_pred_error[f'LOCO-AdaMP'][f'{simu_type}'][f'{ml_name}'])
            LOCOMP_pred.append(dfs_pred_error[f'LOCO-MP'][f'{simu_type}'][f'{ml_name}'])
            LOCOSplit1_pred.append(dfs_pred_error[f'LOCO-Split0.5'][f'{simu_type}'][f'{ml_name}'])
            LOCOSplit2_pred.append(dfs_pred_error[f'LOCO-Split0.75'][f'{simu_type}'][f'{ml_name}']) 
        
        barWidth = 0.1
        br1 = np.arange(len(ml_names)) 
        br2 = [x + barWidth for x in br1]  
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        br5 = [x + barWidth for x in br4]

        ax[i].bar(br1, FullData_pred, width = barWidth, color= color_list[4], zorder=10, label ='Baseline (Full Data)') 
        ax[i].bar(br2, LOCOAdaMP_pred, width = barWidth, color= color_list[0], zorder=10, label ='AdaMP')
        ax[i].bar(br3, LOCOMP_pred, width = barWidth, color= color_list[1], zorder=10, label ='MP')
        ax[i].bar(br4, LOCOSplit1_pred, width = barWidth, color= color_list[2], zorder=10, label ='Baseline (50% Data)')
        ax[i].bar(br5, LOCOSplit2_pred, width = barWidth, color= color_list[3], zorder=10, label ='Baseline (75% Data)')
        ax[i].set_xticks([r + 0.5*barWidth for r in range(len(ml_names))], ml_names)
        ax[i].set_xlabel('ML Model')
        ax[i].set_ylabel('Prediction Error')
        ax[i].set_title(f'{simu_type} Model')
        ax[i].grid(True, zorder=0, alpha=0.5, linestyle='--')
        ax[i].set_xticklabels(ml_names)
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels,bbox_to_anchor=(1.8, 0.5),loc='right')   
    plt.tight_layout()
    plt.savefig(f'{figure_path}/PredError_Comparison_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

### PredCI Figure for LOCO-AdaMp
def fig_pred_ci_LOCOAdaMP_all(M, simu_types, ml_names, indep_delta, target_inds, dfs_adap_error, dfs_adap_samp_prob, dfs_adap_target, dfs_adap_ci, figure_path = '.'):
    # Prediction Error
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax = ax.flatten()
    for i,simu_type in enumerate(simu_types):
        for j,ml_name in enumerate(ml_names):
            error = dfs_adap_error[f'{simu_type}'][f'{ml_name}']
            ax[i].plot(error['epoch'],error['pred_err'],linestyle='-', marker='o', label = f'{ml_name}')
        ax[i].set_xticks(error['epoch'])
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Prediction Error')
        ax[i].set_ylim([-0.03, 1.03])
        ax[i].set_title(f'{simu_type} Model')
        ax[i].grid(True, zorder=0, alpha=0.5, linestyle='--')
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels,bbox_to_anchor=(1.6, 0.5),loc='right')     
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_PredError_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Training Error
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax = ax.flatten()
    for i,simu_type in enumerate(simu_types):
        for j,ml_name in enumerate(ml_names):
            error = dfs_adap_error[f'{simu_type}'][f'{ml_name}']       
            ax[i].plot(error['epoch'],error['train_err'],linestyle='-', marker='o', label = f'{ml_name}')
        ax[i].set_xticks(error['epoch'])
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Traning Error')
        ax[i].set_ylim([-0.03, 1.03])
        ax[i].set_title(f'{simu_type} Model')
        ax[i].grid(True, zorder=0, alpha=0.5, linestyle='--')
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels,bbox_to_anchor=(1.6, 0.5),loc='right')
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_TrainError_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Sampling Probability
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for i,simu_type in enumerate(simu_types):   
        for j,ml_name in enumerate(ml_names):
            ax = axes[i,j]
            ax.axhline(y=indep_delta, color='red', linestyle='--', linewidth=1, label='Delta Line (0.9)')
            for b in range(len(error['epoch'])):            
                sampling_prob = dfs_adap_samp_prob[f'{simu_type}'][f'{ml_name}'][b]
                ax.plot(sampling_prob['feature'],sampling_prob['indep_samp_prob'],linestyle='-', marker='o', label = f'Epoch {b+1}')
            ax.set_xticks(sampling_prob['feature'])
            ax.set_xlabel('Feature')
            ax.set_ylabel('Sampling Probability')
            ax.set_title(f'{simu_type} | {ml_name}')
            ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    handles, labels = axes[1,2].get_legend_handles_labels()
    axes[1,2].legend(handles, labels,bbox_to_anchor=(1.65, 0.5),loc='right')  
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_SampProb_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Target
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for i,simu_type in enumerate(simu_types):   
        for j,ml_name in enumerate(ml_names):
            ax = axes[i,j]
            for tar in target_inds:            
                target = dfs_adap_target[f'{simu_type}'][f'{ml_name}'][tar]
                ax.plot(target['epoch'],target['target'],linestyle='-', marker='o', label = f'Target{tar+1}')
            ax.set_xticks(target['epoch'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Target')
            ax.set_title(f'{simu_type} | {ml_name}')
            ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
    handles, labels = axes[1,2].get_legend_handles_labels()
    axes[1,2].legend(handles, labels,bbox_to_anchor=(1.4, 0.5),loc='right')  
    plt.tight_layout()
    plt.savefig(f'{figure_path}/AdaMP_Target_M{M}.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Confidence Interval
    for i,simu_type in enumerate(simu_types):  
        for j,ml_name in enumerate(ml_names):        
            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            ax = ax.flatten()      
            for b in range(len(error['epoch'])):
                ci = dfs_adap_ci[f'{simu_type}'][f'{ml_name}'][b]
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



### Coverage
def fig_cover_LOCOAdaMP_all(M, simu_types, ml_names, dfs_coverage, figure_path = '.'):
    # Coverage
    def fig_coverage(coverage_s, coverage_n, err_s, err_n, Coverage):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        for i,simu_type in enumerate(simu_types):   
            for j,ml_name in enumerate(ml_names):
                ax = axes[i,j]
                ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='0.9 Line')       
                coverage = dfs_coverage[f'{simu_type}'][f'{ml_name}']
                ax.errorbar(x=coverage['N'], y=coverage[f'{coverage_s}'], yerr=coverage[f'{err_s}'], linestyle="-", capsize=3, marker="o", label = "Signal")
                ax.errorbar(x=coverage['N'], y=coverage[f'{coverage_n}'], yerr=coverage[f'{err_n}'], linestyle="-", capsize=3, marker="o", label = "Noise")
                ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
                ax.set_xlabel('N')
                ax.set_xticks(coverage['N'])
                ax.set_ylim([-0.03, 1.03])
                ax.set_ylabel('Coverage')
                ax.set_title(f'{simu_type} | {ml_name}')
        handles, labels = axes[1,2].get_legend_handles_labels()
        axes[1,2].legend(handles, labels,bbox_to_anchor=(1.42, 0.5),loc='right')  
        plt.tight_layout()
        plt.savefig(f'{figure_path}/{Coverage}_M{M}.png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    fig_coverage('coverage_s', 'coverage_n', 'err_s', 'err_n','Coverage')
    fig_coverage('coverage_adj_s', 'coverage_adj_n', 'err_adj_s', 'err_adj_n','Coverage_Adj')

    # Interval Width, Target, Variance
    def fig_cover_item(item_name, feature_name, ylabel_name, save_name):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        for i,simu_type in enumerate(simu_types):   
            for j,ml_name in enumerate(ml_names):
                ax = axes[i,j]
                coverage = dfs_coverage[f'{simu_type}'][f'{ml_name}']
                ax.plot(coverage['N'],coverage[f'{item_name}'],linestyle='-', marker='o', label = f'{feature_name}')
                ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
                ax.set_xlabel('N')
                ax.set_xticks(coverage['N'])
                ax.set_ylabel(f'{ylabel_name}')
                ax.set_title(f'{simu_type} | {ml_name}')
        handles, labels = axes[1,2].get_legend_handles_labels()
        axes[1,2].legend(handles, labels,bbox_to_anchor=(1.42, 0.5),loc='right')  
        plt.tight_layout()
        plt.savefig(f'{figure_path}/{save_name}_M{M}.png', dpi=300, bbox_inches="tight")
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



### Power
def fig_power_LOCO(M, simu_types, ml_names, loco_meths, dfs_power, dfs_target, figure_path = '.'):
    # Histogram
    def fig_target_z(loco_meth):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        for i,simu_type in enumerate(simu_types):
            for j,ml_name in enumerate(ml_names):
                ax = axes[i,j]
                ax.hist(dfs_target[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}'], bins=15, label=["Target", "Z"])
                ax.set_xlabel('Target/Z')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{simu_type} | {ml_name}')
                ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
        handles, labels = axes[1,2].get_legend_handles_labels()
        axes[1,2].legend(handles, labels,bbox_to_anchor=(1.41, 0.5),loc='right')
        plt.tight_layout()
        plt.savefig(f'{figure_path}/TargetZ_Hist_{loco_meth}_M{M}.png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    for loco_meth in loco_meths:
        fig_target_z(loco_meth)
    
    # Power Comparison
    # Power
    def get_fig_power(ci = True):
        color_list = ['#d62728', '#2ca02c','#ff7f0e','#1f77b4']#red, green, orange, blue
        fig, axes = plt.subplots(3, 3, figsize=(12.5, 8))

        for i,simu_type in enumerate(simu_types):   
            for j,ml_name in enumerate(ml_names):
                ax = axes[i,j]
                for l, loco_meth in enumerate(loco_meths):  
                    zo = 1
                    if loco_meth == "LOCO-AdaMP":
                        zo = 10                  
                    power = dfs_power[f'{loco_meth}'][f'{simu_type}'][f'{ml_name}']
                    if ci:
                        ax.errorbar(x=power['beta'],y=power['power'],yerr=power['err'],color = color_list[l],linestyle='-', marker='o', label = f'{loco_meth}', zorder=zo)
                    else:
                        ax.plot(power['beta'],power['power'],color = color_list[l],linestyle='-', marker='o', label = f'{loco_meth}', zorder=zo)
                ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='0.9 Line')
                ax.set_xlabel('Beta')
                ax.set_xticks(power['beta'])
                ax.set_ylim([-0.03, 1.03])
                ax.set_ylabel('Power')
                ax.set_title(f'{simu_type} | {ml_name}')
                ax.grid(True, zorder=0, alpha=0.5, linestyle='--')
        handles, labels = axes[1,2].get_legend_handles_labels()
        axes[1,2].legend(handles, labels,bbox_to_anchor=(1.58, 0.5),loc='right')
        plt.tight_layout()
        if ci:
            plt.savefig(f'{figure_path}/PowerCI_Comparison_M{M}.png', dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f'{figure_path}/Power_Comparison_M{M}.png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    get_fig_power(ci = True)
    get_fig_power(ci = False)

