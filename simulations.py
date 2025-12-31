import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# X, Y, and N are for training data
# X1, Y1, and N1 are for test data

def simu_independent(N,M,N1,beta):
    X =  np.random.normal(0, 1, (N, M))
    X1 =  np.random.normal(0, 1, (N1, M))

    beta = np.append(np.array(beta), np.array([0]*(M-len(beta))))
    Y = np.dot(X,beta)+np.random.normal(0, 1, N)
    Y1 = np.dot(X1,beta)+np.random.normal(0, 1, N1)

    scaler = StandardScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1))
    Y1 = scaler.transform(Y1.reshape(-1,1))
    Y = Y.reshape(-1)
    Y1 = Y1.reshape(-1)

    return [X,Y,X1,Y1]

def simu_correlated(N,M,N1,beta,rho):
    mu = [0]*M
    cov = [[0]*M for _ in range(M)]
    cov[rho[1]][rho[2]]=rho[0]
    cov[rho[2]][rho[1]]=rho[0]
    cov=np.array(cov)
    np.fill_diagonal(cov, 1)

    X = np.random.multivariate_normal(mu, cov, N)
    X1 =  np.random.multivariate_normal(mu, cov, N1)

    beta = np.append(np.array(beta), np.array([0]*(M-len(beta))))
    Y = np.dot(X,beta)+np.random.normal(0, 1, N)
    Y1 = np.dot(X1,beta)+np.random.normal(0, 1, N1)

    scaler = StandardScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1))
    Y1 = scaler.transform(Y1.reshape(-1,1))
    Y = Y.reshape(-1)
    Y1 = Y1.reshape(-1)

    return [X,Y,X1,Y1]

def simu_nonlinear(N,M,N1,beta):
    X =  np.random.normal(0, 1, (N, M))
    X1 =  np.random.normal(0, 1, (N1, M)) 

    X_0 = (X[:,0]<2 &(X[:,0]>-2))
    X_1 = (X[:,1]>0) * X[:,1]
    X_2 = (X[:,2]<-1) * X[:,2]
    X_3 = (X[:,3]>0 &(X[:,3]<3))
    X_4 = (np.sign(X[:,4]))
    X1_0 = (X1[:,0]<2 &(X1[:,0]>-2))
    X1_1 = (X1[:,1]>0) * X1[:,1]
    X1_2 = (X1[:,2]<-1) * X1[:,2]
    X1_3 = (X1[:,3]>0 &(X1[:,3]<3))
    X1_4 = (np.sign(X1[:,4]))

    Y = beta[0]*(X_0/np.std(X_0))+beta[1]*(X_1/np.std(X_1))+beta[2]*(X_2/np.std(X_2))+beta[3]*(X_3/np.std(X_3))+ beta[4]*(X_4/np.std(X_4))+np.random.normal(0, 1,N)
    Y1 = beta[0]*(X1_0/np.std(X1_0))+beta[1]*(X1_1/np.std(X1_1))+beta[2]*(X1_2/np.std(X1_2))+beta[3]*(X1_3/np.std(X1_3))+ beta[4]*(X1_4/np.std(X1_4))+np.random.normal(0, 1,N1) 
    
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y.reshape(-1,1))
    Y1 = scaler.transform(Y1.reshape(-1,1))
    Y = Y.reshape(-1)
    Y1 = Y1.reshape(-1)

    return [X,Y,X1,Y1]


### Data Generation

def data_generate_pred(M, N, N1, beta, rho = 0.9, simu_type = 'all', path = '.'):
    if simu_type == 'Independent':
        f = open(f'{path}/sim_Independent_M{M}.pkl','wb')
        pickle.dump(simu_independent(N,M,N1,beta),f)
        f.close()
    if simu_type == 'Correlated':
        f = open(f'{path}/sim_Correlated_M{M}.pkl','wb')
        pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
        f.close()
    if simu_type == 'Nonlinear':
        f = open(f'{path}/sim_Nonlinear_M{M}.pkl','wb')
        pickle.dump(simu_nonlinear(N,M,N1,beta),f)
        f.close()
    if simu_type == 'all':
        f = open(f'{path}/sim_Independent_M{M}.pkl','wb')
        pickle.dump(simu_independent(N,M,N1,beta),f)
        f.close()
        f = open(f'{path}/sim_Correlated_M{M}.pkl','wb')
        pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
        f.close()
        f = open(f'{path}/sim_Nonlinear_M{M}.pkl','wb')
        pickle.dump(simu_nonlinear(N,M,N1,beta),f)
        f.close()


def data_genarate_cover(M, Ns, N1, beta, rho = 0.9, rep = 100, simu_type = 'all', path = '.'):
    for N in Ns:
        for i in range(rep):
            if simu_type == 'Independent':
                f = open(f'{path}/sim_Independent_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_independent(N,M,N1,beta),f)
                f.close()
            if simu_type == 'Correlated':
                f = open(f'{path}/sim_Correlated_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
                f.close()
            if simu_type == 'Nonlinear':
                f = open(f'{path}/sim_Nonlinear_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_nonlinear(N,M,N1,beta),f)
                f.close()
            if simu_type == 'all':
                f = open(f'{path}/sim_Independent_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_independent(N,M,N1,beta),f)
                f.close()
                f = open(f'{path}/sim_Correlated_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
                f.close()
                f = open(f'{path}/sim_Nonlinear_M{M}_N{N}_{i}.pkl','wb')
                pickle.dump(simu_nonlinear(N,M,N1,beta),f)
                f.close()


def data_generate_power(M, N, N1, betas, rho = 0.9, rep = 100, beta_ind = 4, simu_type = 'all', path = '.'):
    for beta in betas:
        for i in range(rep):
            if simu_type == 'Independent':
                f = open(f'{path}/sim_Independent_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_independent(N,M,N1,beta),f)
                f.close()
            if simu_type == 'Correlated':
                f = open(f'{path}/sim_Correlated_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
                f.close()
            if simu_type == 'Nonlinear':
                f = open(f'{path}/sim_Nonlinear_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_nonlinear(N,M,N1,beta),f)
                f.close()
            if simu_type == 'all':
                f = open(f'{path}/sim_Independent_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_independent(N,M,N1,beta),f)
                f.close()
                f = open(f'{path}/sim_Correlated_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_correlated(N,M,N1,beta,rho),f)
                f.close()
                f = open(f'{path}/sim_Nonlinear_Beta{beta[beta_ind]}_M{M}_{i}.pkl','wb')
                pickle.dump(simu_nonlinear(N,M,N1,beta),f)
                f.close()


