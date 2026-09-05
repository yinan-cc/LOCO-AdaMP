import os
import argparse
from data_generate import *
from loco_methods import *
from ml_models import *
from utils import *
from pred_ci import *
from coverage import *
from power import *
from figure import *
import matplotlib.pyplot as plt

def run_pred_ci(M, N, N1, beta, rho, loco_meths, simu_type, ml_name, rep, ratios, alpha, indep_delta, target_inds, n_features, n_jobs, lossfunc, seed0):   

    simulation_predci = f"Simulation_Data/PredCI_{M}"

    if not os.path.exists(simulation_predci):
        os.mkdir(simulation_predci)
    
    file_path = os.path.join(f"Simulation_Data/PredCI_{M}", f"sim_{simu_type}_M{M}.pkl")
    if not os.path.exists(file_path):
        data_generate_pred(M, N, N1, beta, rho, rep, simu_type, seed0, path = simulation_predci)

    output_predci = "Output_Data/PredCI"
    if not os.path.exists(output_predci):
        os.mkdir(output_predci)

    figure_predci = "Figure/PredCI"
    if not os.path.exists(figure_predci):
        os.mkdir(figure_predci)

    figure_ci = "Figure/PredCI/AdaCI"
    if not os.path.exists(figure_ci):
        os.mkdir(figure_ci)
    
    inf_pred_ci(M, N, simu_type, ml_name, rep, ratios, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_predci, output_path = output_predci,seed0=seed0)
    res_pred_ci(M, loco_meths, simu_type, ml_name, rep, ratios, alpha, target_inds, n_features, lossfunc, simu_path = simulation_predci, output_path = output_predci,seed0=seed0)
    fig_pred_ci(M, N, loco_meths, simu_type, ml_name, indep_delta, target_inds, lossfunc, output_path = output_predci, figure_path = figure_predci)


def run_coverage(M, Ns, N1, beta, rho, simu_type, ml_name, rep, alpha, indep_delta, signal, noise, n_jobs, lossfunc,seed0):

    simulation_cover = f"Simulation_Data/Coverage_{M}"
    if not os.path.exists(simulation_cover):
        os.mkdir(simulation_cover)
    
    file_path = os.path.join(f"Simulation_Data/Coverage_{M}", f"sim_{simu_type}_M{M}_N{Ns[-1]}_rep.pkl")
    if not os.path.exists(file_path):
        data_genarate_cover(M, Ns, N1, beta, rho, rep, simu_type, seed0, path = simulation_cover)

    output_cover = "Output_Data/Coverage"
    if not os.path.exists(output_cover):
        os.mkdir(output_cover)

    figure_cover = "Figure/Coverage"
    if not os.path.exists(figure_cover):
        os.mkdir(figure_cover)

    inf_coverage(M, Ns, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_cover, output_path = output_cover,seed0=seed0)
    res_cover(M, Ns, simu_type, ml_name, rep, alpha, signal, noise, lossfunc, output_path = output_cover)
    fig_cover(M, simu_type, ml_name, lossfunc, output_path = output_cover, figure_path = figure_cover)


def run_power(M, N, N1, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, n_jobs, lossfunc,seed0):

    simulation_power = f"Simulation_Data/Power_{M}"
    if not os.path.exists(simulation_power):
        os.mkdir(simulation_power)
    
    file_path = os.path.join(f"Simulation_Data/Power_{M}", f"sim_{simu_type}_Beta{betas[-1][beta_ind]}_M{M}_rep.pkl")
    if not os.path.exists(file_path):
        data_generate_power(M, N, N1, betas, rho, rep, beta_ind, simu_type, seed0, path = simulation_power)

    output_power = "Output_Data/Power"
    if not os.path.exists(output_power):
        os.mkdir(output_power)

    figure_power = "Figure/Power"
    if not os.path.exists(figure_power):
        os.mkdir(figure_power)

    inf_power(M, N, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_power, output_path = output_power,seed0=seed0)
    res_power(M, loco_meths, simu_type, ml_name, betas, rep, alpha, beta_ind, pvalue_thre, lossfunc, output_path = output_power)
    fig_power(M, loco_meths, simu_type, ml_name, lossfunc, output_path = output_power, figure_path = figure_power)



def run_inference(inf, M, N, Ns, N1, beta, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features, n_jobs, lossfunc,seed0):

    simulation = "Simulation_Data"
    if not os.path.exists(simulation):
        os.mkdir(simulation)
    output = "Output_Data"
    if not os.path.exists(output):
        os.mkdir(output)
    figure = "Figure"
    if not os.path.exists(figure):
        os.mkdir(figure)

    if inf == 'predci':
        run_pred_ci(M, N, N1, beta, rho, loco_meths, simu_type, ml_name, rep, ratios, alpha, indep_delta, target_inds, n_features, n_jobs, lossfunc,seed0)
    elif inf == 'coverage':
        run_coverage(M, Ns, N1, beta, rho, simu_type, ml_name, rep, alpha, indep_delta, signal, noise, n_jobs, lossfunc,seed0)
    elif inf == 'power':
        run_power(M, N, N1, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, n_jobs, lossfunc,seed0)

def run_simulation(M, N, Ns, N1, betas, beta, rho, rep, beta_ind, simu_type, seed0):
    simulation = "Simulation_Data"
    if not os.path.exists(simulation):
        os.mkdir(simulation)

    simulation_predci = f"Simulation_Data/PredCI_{M}"
    if not os.path.exists(simulation_predci):
        os.mkdir(simulation_predci)
        data_generate_pred(M, N, N1, beta, rho, rep , simu_type, seed0, path = simulation_predci)

    simulation_cover = f"Simulation_Data/Coverage_{M}"
    if not os.path.exists(simulation_cover):
        os.mkdir(simulation_cover)
        data_genarate_cover(M, Ns, N1, beta, rho, rep , simu_type, seed0, path = simulation_cover)

    simulation_power = f"Simulation_Data/Power_{M}"
    if not os.path.exists(simulation_power):
        os.mkdir(simulation_power)
        data_generate_power(M, N, N1, betas, rho, rep, beta_ind, simu_type, seed0, path = simulation_power)

def run_figure(inf, M, N, loco_meths, indep_delta, target_inds, lossfunc):
    simu_types = ["Independent", "Correlated", "Nonlinear"]
    ml_names = ["Ridge", "RandomForest", "KernelRidge"]

    if inf == 'predci':
        figure_predci = "Figure/Figure_All/PredCI"
        if not os.path.exists(figure_predci):
            os.mkdir(figure_predci)
        figure_ci = "Figure/Figure_All/PredCI/AdaCI"
        if not os.path.exists(figure_ci):
            os.mkdir(figure_ci)
        fig_pred_ci_all(M, N, loco_meths, simu_types, ml_names, indep_delta, target_inds, lossfunc, output_path = "Output_Data/PredCI", figure_path = figure_predci)
    
    if inf == 'coverage':
        figure_cover = "Figure/Figure_All/Coverage"
        if not os.path.exists(figure_cover):
            os.mkdir(figure_cover)
        fig_cover_all(M, simu_types, ml_names, lossfunc, output_path = "Output_Data/Coverage", figure_path = figure_cover)
    
    if inf == 'power':
        figure_power = "Figure/Figure_All/Power"
        if not os.path.exists(figure_power):
            os.mkdir(figure_power)
        fig_power_all(M, loco_meths, simu_types, ml_names, lossfunc, output_path = "Output_Data/Power", figure_path = figure_power)


def main(N, Ns, N1, beta, rho, loco_meths, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features):
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    simu_data = "no"
    fig_only = "no"
    
    dim = "low" #dimension: choices=["low", "high"]
    inf = "predci" #inference: choices=["predci", "coverage", "power"]
    simu = "independent" #data generating model: choices=["independent", "correlated", "nonlinear"]
    ml = "ridge" #ml model: choices=["ridge", "randomforest", "kernelridge"]
    lossfunc = "sq" #loss function: choices=["sq", "abs"]

    M = get_M(dim)
    simu_type = format_simu_type(simu)
    ml_name = format_ml_name(ml)

    ##########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=str, default="low", choices=["low", "high"])
    parser.add_argument("--inf", type=str, default="predci", choices=["predci", "coverage", "power"])
    parser.add_argument("--simu", type=str, default="independent", choices=["independent", "correlated", "nonlinear"])
    parser.add_argument("--ml", type=str, default="ridge", choices=["ridge", "randomforest", "kernelridge"])
    parser.add_argument("--loss", type=str, default="sq", choices=["sq", "abs"])
    parser.add_argument("--simudata", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--figonly", type=str, default="no", choices=["yes", "no"])

    args = parser.parse_args()
    M = get_M(args.dim)
    inf = args.inf
    simu_type = format_simu_type(args.simu)
    ml_name = format_ml_name(args.ml)
    lossfunc = args.loss
    simu_data = args.simudata
    fig_only = args.figonly

    get_parameter(M, N) ############

    seed0=100

    if simu_data == "yes":
        run_simulation(M, N, Ns, N1, betas, beta, rho, rep, beta_ind, simu_type = 'all', seed0 = seed0)
    if simu_data == "no":
        if fig_only == "yes":
            run_figure(inf, M, N, loco_meths, indep_delta, target_inds, lossfunc)
        if fig_only == "no":
            run_inference(inf, M, N, Ns, N1, beta, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features, n_jobs, lossfunc,seed0=seed0)

if __name__ == "__main__":
    ### Parameter setting
    N = 200
    Ns = [200, 500, 1000, 2000] 
    N1 = 10000
    rho = [0.9, 4, 5] # the correlation coefficient between the 5th and the 6th features is 0.9
    rep = 100 

    beta = [2.5, 2.1, 1.7, 1.3, 1]
    beta_ind = 4 # the fifth feature
    beta_0 = [2.5, 2.1, 1.7, 1.3]
    beta_1 = [0, 0.4, 1, 2.5, 5]
    betas = get_betas(beta_1, beta_0)

    alpha= 0.1
    indep_delta = 0.9
    ratios = [0.5,0.75] # for LOCO-Split
    pvalue_thre = 0.1

    loco_meths = ["LOCO-AdaMP","LOCO-MP","LOCO-Split0.5","LOCO-Split0.75"]

    n_features = 8 # use the first 8 features to build CI
    signal = 4 # signal: choose the 5th feature
    noise = 5 # noise: choose the 6th feature
    target_inds = [0, 4, 5] # choose the 1st, 5th, and 6th features

    plt.rcParams['axes.titlesize'] = 12.5
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    #color_list = ["#d62728", '#2ca02c','#ff7f0e','#1f77b4', '#9467bd', "#8c564b"]


    simulation = "Simulation_Data"
    if not os.path.exists(simulation):
        os.mkdir(simulation)
    output = "Output_Data"
    if not os.path.exists(output):
        os.mkdir(output)
    figure = "Figure"
    if not os.path.exists(figure):
        os.mkdir(figure)

    output_predci = "Output_Data/PredCI"
    if not os.path.exists(output_predci):
        os.mkdir(output_predci)

    figure_predci = "Figure/PredCI"
    if not os.path.exists(figure_predci):
        os.mkdir(figure_predci)

    figure_ci = "Figure/PredCI/AdaCI"
    if not os.path.exists(figure_ci):
        os.mkdir(figure_ci)

    output_cover = "Output_Data/Coverage"
    if not os.path.exists(output_cover):
        os.mkdir(output_cover)

    figure_cover = "Figure/Coverage"
    if not os.path.exists(figure_cover):
        os.mkdir(figure_cover)

    output_power = "Output_Data/Power"
    if not os.path.exists(output_power):
        os.mkdir(output_power)

    figure_power = "Figure/Power"
    if not os.path.exists(figure_power):
        os.mkdir(figure_power)
    
    figure_all = "Figure/Figure_All"
    if not os.path.exists(figure_all):
        os.mkdir(figure_all)

    main(N, Ns, N1, beta, rho, loco_meths, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features)
