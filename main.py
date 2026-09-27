import os
import shutil
import argparse
from data_generate import *
from loco_methods import *
from ml_models import *
from utils import *
from pred_ci import *
from coverage import *
from power import *
import matplotlib.pyplot as plt

def run_pred_ci(M, N, N1, beta, rho, loco_meths, simu_type, ml_name, rep, ratios, alpha, indep_delta, target_inds, n_features, n_jobs, lossfunc, seed0):   

    simulation_path = f"Simulation_Data_PCI{M}"
    os.mkdir(simulation_path)
    data_generate_pred(M, N, N1, beta, rho, rep, simu_type, seed0, path = simulation_path)
    output_path = f"Output_Data_PCI{M}"
    os.mkdir(output_path)
    
    inf_pred_ci(M, N, simu_type, ml_name, rep, ratios, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_path, output_path = output_path,seed0=seed0)
    res_pred_ci(M, loco_meths, simu_type, ml_name, rep, ratios, alpha, target_inds, n_features, lossfunc, simu_path = simulation_path, output_path = output_path,seed0=seed0)
    fig_pred_ci(M, N, loco_meths, simu_type, ml_name, indep_delta, target_inds, lossfunc, output_path = output_path)

    shutil.rmtree(simulation_path)
    shutil.rmtree(output_path)

def run_coverage(M, Ns, N1, beta, rho, simu_type, ml_name, rep, alpha, indep_delta, signal, noise, n_jobs, lossfunc,seed0):

    simulation_path = f"Simulation_Data_C{M}"
    os.mkdir(simulation_path)
    data_genarate_cover(M, Ns, N1, beta, rho, rep, simu_type, seed0, path = simulation_path)
    output_path = f"Output_Data_C{M}"
    os.mkdir(output_path)

    inf_coverage(M, Ns, simu_type, ml_name, rep, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_path, output_path = output_path,seed0=seed0)
    res_cover(M, Ns, simu_type, ml_name, rep, alpha, signal, noise, lossfunc, output_path = output_path)
    fig_cover(M, simu_type, ml_name, lossfunc, output_path = output_path)

    shutil.rmtree(simulation_path)
    shutil.rmtree(output_path)


def run_power(M, N, N1, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, n_jobs, lossfunc,seed0):

    simulation_path = f"Simulation_Data_P{M}"
    os.mkdir(simulation_path)
    data_generate_power(M, N, N1, betas, rho, rep, beta_ind, simu_type, seed0, path = simulation_path)
    output_path = f"Output_Data_P{M}"
    os.mkdir(output_path)

    inf_power(M, N, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, indep_delta, n_jobs, lossfunc, simu_path = simulation_path, output_path = output_path, seed0=seed0)
    res_power(M, loco_meths, simu_type, ml_name, betas, rep, alpha, beta_ind, pvalue_thre, lossfunc, output_path = output_path)
    fig_power(M, loco_meths, simu_type, ml_name, lossfunc, output_path = output_path)

    shutil.rmtree(simulation_path)
    shutil.rmtree(output_path)

def run_inference(inf, M, N, Ns, N1, beta, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features, n_jobs, lossfunc,seed0):

    plt.rcParams['axes.titlesize'] = 12.5
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    if inf == 'predci':
        run_pred_ci(M, N, N1, beta, rho, loco_meths, simu_type, ml_name, rep, ratios, alpha, indep_delta, target_inds, n_features, n_jobs, lossfunc,seed0)
    elif inf == 'coverage':
        run_coverage(M, Ns, N1, beta, rho, simu_type, ml_name, rep, alpha, indep_delta, signal, noise, n_jobs, lossfunc,seed0)
    elif inf == 'power':
        run_power(M, N, N1, rho, loco_meths, simu_type, ml_name, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, n_jobs, lossfunc,seed0)

def main(N, Ns, N1, beta, rho, loco_meths, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features):
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    
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

    args = parser.parse_args()
    M = get_M(args.dim)
    inf = args.inf
    simu_type = format_simu_type(args.simu)
    ml_name = format_ml_name(args.ml)
    lossfunc = args.loss

    seed0=100

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

    main(N, Ns, N1, beta, rho, loco_meths, ratios, betas, rep, beta_ind, alpha, pvalue_thre, indep_delta, target_inds, signal, noise, n_features)
