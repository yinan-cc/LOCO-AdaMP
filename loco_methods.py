import numpy as np
from scipy.stats import *

### Inference 
def ztest(z,alpha,MM=1,bonf_correct=False):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
    l = len(z)
    s = np.std(z)
    if s==0:
        return [0,0,0,0]
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/s*np.sqrt(l))

    pval2 = 2*(1-norm.cdf(np.abs(m/s*np.sqrt(l))))

    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
        
    q = norm.ppf(1-alpha/2)
    left  = m - q*s/np.sqrt(l)
    right = m + q*s/np.sqrt(l)

    return [pval1,pval2, left,right]

def ztest_target(z, target, alpha,MM=1,bonf_correct=False):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
    l_z = len(z)
    l_t = len(target)
    var_z = np.var(z)
    var_t = np.var(target)
    if s==0:
        return [0,0,0,0]
    m = np.mean(z)
    
    # Apply Bonferroni correction for M tests
    if bonf_correct:
        alpha = alpha/MM
        
    q = norm.ppf(1-alpha/2)
    left  = m - q*np.sqrt(var_z/l_z + var_t/l_t)
    right = m + q*np.sqrt(var_z/l_z + var_t/l_t)

    return [left,right]

def loss_func(a1, a2, func = "sq"):
    if func == "sq":
        loss = np.square(a1 - a2)
    if func == "abs":
        loss = np.abs(a1 - a2)
    return loss

### Method: LOCO-AdaMP 

def indep_sampling_probability(z, m, indep_delta = 0.9, c = 0.1):    
    #z = errorLOCO - errorLOO
    prob = np.zeros(len(z))
    for j in range(len(z)):
        prob[j] = z[j].mean()
    prob = prob - min(prob) + c/len(z)
    prob_sort = np.sort(prob)[::-1]
    total_sum = sum(prob_sort)
    running_sum = 0
    target_index = 0
    if total_sum > (m/indep_delta) * prob_sort[0]:
        target_index = -1
    else:
        for j in range(len(prob_sort)):
            running_sum += prob_sort[j]
            if (total_sum - running_sum > (m/indep_delta - j - 1) * prob_sort[j+1]) and (m/indep_delta - j - 1 > 0):
                target_index = j
                break
            elif j == len(prob_sort) - 1: print("Can not find target index j")
    prob_star = sum(prob_sort[target_index+1:])/(m/indep_delta-(target_index+1))
    t_star = min(prob_star, prob_sort[0])
    weight = np.array([min(prob_j, t_star) for prob_j in prob])
    indep_samp_prob = m*weight/sum(weight)
    return indep_samp_prob

def indept_sample_array(probability):
    sampled_indices = []
    for i in range(len(probability)):
        rn = np.random.rand()
        if  rn < probability[i]: # Probability of including element i
            sampled_indices.append(i)
    return sampled_indices

def indep_buildMP(X, Y, n, indep_samp_prob):
    N = len(X) 
    # index of minipatch
    idx_I = np.sort(np.random.choice(N, size=n, replace=False))
    idx_F = np.array([])
    while len(idx_F) == 0:
        idx_F = np.sort(indept_sample_array(indep_samp_prob))
    # record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]

def indep_predictMP(X, Y, X1, n, B, fit_funct, indep_samp_prob):
    N = len(X)
    M = len(X[0])
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B): 
        [idx_I,idx_F,x_mp,y_mp] = indep_buildMP(X, Y, n, indep_samp_prob)
        predictions.append((fit_funct().fit(x_mp,y_mp)).predict(X1[:, idx_F]))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def LOCOAdaMP(X, Y, X1, Y1, n, m, Kb, fit_funct, indep_samp_prob, alpha, lossfunc, indep_delta, selected_features = [], bonf=False):
    ress={}
    for ite in range(len(Kb)):
        B = Kb[ite]
        res = LOCOAdaMP_ite(X,Y,X1,Y1, n, B, fit_funct, indep_samp_prob, alpha, lossfunc, selected_features, bonf)
        ress[ite] = res   
        indep_samp_prob = indep_sampling_probability(res['z'], m, indep_delta)
    return ress

def LOCOAdaMP_ite(X, Y, X1, Y1, n, B, fit_funct, indep_samp_prob, alpha, lossfunc, selected_features = [], bonf=False):
    N=len(X)
    M = len(X[0])

    [predictions,in_mp_obs,in_mp_feature]= indep_predictMP(X,Y,np.vstack((X,X1)), n, B, fit_funct, indep_samp_prob)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]
      
    # Re-fit after dropping each feature
    resids_LOO,resids_LOCO = np.zeros(N),np.zeros(N)

    ### Find LOO
    for i in range(N):
        ## find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
   
        if len(b_keep)>0:
            resids_LOO[i]= loss_func(Y[i], predictions_train[b_keep,i].mean(), lossfunc)
    
    ### FIND LOCO
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z = np.zeros((len(ff),4))
    z={}
    resids_LOCOs ={}

    for idd,j in enumerate(ff):
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= loss_func(Y[i], predictions_train[b_keep_f,i].mean(), lossfunc)
            
        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()
             
        if len(z)==0:
            inf_z[idd] = [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM = len(ff),bonf_correct =bonf)

    uhat_test=predictions_test.mean(0)
    mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    uhat1_test=predictions_test[mps1,:].mean(0)
    uhat2_test=predictions_test[mps2,:].mean(0)

    var,target,err1,err2 = {},{},{},{}
    err11,err12,err21,err22={},{},{},{}
    stability_err,stability={},{}
    adj_ci = {}

    for idd,j in enumerate(ff): ## ff include feature of interest
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0) ## TEST ERROR WITHOUT J
        
        b_keep_f1 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
        uhat1_j_test=predictions_test[b_keep_f1,:].mean(0)
        b_keep_f2 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
        uhat2_j_test=predictions_test[b_keep_f2,:].mean(0)
    
        err1[idd] =  loss_func(Y1, uhat_j_test, lossfunc)
        err2[idd] = loss_func(Y1, uhat_test, lossfunc)
        target[idd] = np.mean(err1[idd]-err2[idd])
        var[idd] = np.var(err1[idd]-err2[idd])
        
        err11[idd] = loss_func(Y1, uhat1_j_test, lossfunc)
        err12[idd] = loss_func(Y1, uhat1_test, lossfunc)
        err21[idd] = loss_func(Y1, uhat2_j_test, lossfunc)
        err22[idd] = loss_func(Y1, uhat2_test, lossfunc)
        stability_err[idd] = err11[idd] - err12[idd] - err21[idd] + err22[idd]
        stability[idd]= np.std(stability_err[idd])

        adj_ci[idd] = ztest_target(z[idd], err1[idd]-err2[idd], alpha,MM=len(ff),bonf_correct=bonf)

    ### result
    res= {}
    res['indep_samp_prob'] = indep_samp_prob
    res['err2'] = err2
    res['resids_LOO']=resids_LOO
    res['inf']=inf_z
    res['z']=z
    res['target']=target
    res['variance']=var
    res['adj_ci'] = adj_ci
    res['stability']=stability
    res['resids_LOCO']=resids_LOCOs
    res['err1'] = err1
    res['stability_err'] = stability_err

    return res

### Method: LOCO-MP ###############################################################################################

def buildMP(X,Y,n,m):
    N = len(X)
    M = len(X[0])
    ## index of minipatch
    idx_I = np.sort(np.random.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(np.random.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]

def predictMP(X,Y,X1,n,m,K_locomp,fit_funct):
    N = len(X)
    M = len(X[0])
    in_mp_obs,in_mp_feature = np.zeros((K_locomp,N),dtype=bool),np.zeros((K_locomp,M),dtype=bool)
    predictions=[]
    for b in range(K_locomp):        
        [idx_I,idx_F,x_mp,y_mp] = buildMP(X,Y,n,m)
        predictions.append((fit_funct().fit(x_mp,y_mp)).predict(X1[:, idx_F])) 
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def LOCOMP(X, Y, X1, Y1, n, m, K_locomp, fit_funct, alpha, lossfunc, selected_features=[], bonf=False):
    N=len(X)
    M = len(X[0])

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,np.vstack((X,X1)),n,m,K_locomp,fit_funct)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]
    
    # Re-fit after dropping each feature
    resids_LOO,resids_LOCO = np.zeros(N),np.zeros(N)

    ### Find LOO
    for i in range(N):
        # find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
     
        if len(b_keep)>0:
            resids_LOO[i]= loss_func(Y[i], predictions_train[b_keep,i].mean(), lossfunc)

    ### FIND LOCO
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z = np.zeros((len(ff),4))
    z={}
    resids_LOCOs = {}

    for idd,j in enumerate(ff):
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= loss_func(Y[i], predictions_train[b_keep_f,i].mean(), lossfunc)
        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()

        if len(z)==0:
            inf_z[idd] = [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =bonf)

    uhat_test=predictions_test.mean(0)
    mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    uhat1_test=predictions_test[mps1,:].mean(0)
    uhat2_test=predictions_test[mps2,:].mean(0)
    
    var,target,err1,err2 = {},{},{},{}
    err11,err12,err21,err22={},{},{},{}
    stability_err,stability={},{}

    for idd,j in enumerate(ff): # ff include feature of interest
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0) # TEST ERROR WITHOUT J

        b_keep_f1 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1))) ###
        uhat1_j_test=predictions_test[b_keep_f1,:].mean(0)
        b_keep_f2 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1))) ###
        uhat2_j_test=predictions_test[b_keep_f2,:].mean(0)
        
        err1[idd] =  loss_func(Y1, uhat_j_test, lossfunc)
        err2[idd] = loss_func(Y1, uhat_test, lossfunc)
        target[idd] = np.mean(err1[idd]-err2[idd])
        var[idd] = np.var(err1[idd]-err2[idd])
        
        err11[idd] = loss_func(Y1, uhat1_j_test, lossfunc)
        err12[idd] = loss_func(Y1, uhat1_test, lossfunc)
        err21[idd] = loss_func(Y1, uhat2_j_test, lossfunc)
        err22[idd] = loss_func(Y1, uhat2_test, lossfunc)
        stability_err[idd] = err11[idd] - err12[idd] - err21[idd] + err22[idd]
        stability[idd]= np.std(stability_err[idd])

    ### Result
    res= {}
    res['err2'] = err2
    res['inf']=inf_z
    res['z']=z
    res['target']=target
    res['resids_LOO']=resids_LOO
    res['resids_LOCO']=resids_LOCOs   
    res['variance']=var
    res['err1'] = err1
    res['stability']=stability
    res['stability_err'] = stability_err
   
    return res

### Method: LOCO-Split ############################################################################################

def LOCOSplit(X, Y, X1, Y1, fit_funct, ratio, alpha, lossfunc, selected_features=[], bonf=False):
    N=len(X)
    M = len(X[0])
    
    # uniform sampling of subset of observations
    idx_I = np.sort(np.random.choice(N, size=int(ratio*N), replace=False))
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
   
    prediction = (fit_funct().fit(x_train,y_train)).predict(np.vstack((x_val,X1)))
    predictions  = prediction[:len(y_val)]
    predictions_test = prediction[len(y_val):]

    resids_split = loss_func(y_val, predictions, lossfunc)
    resids_split_test = loss_func(Y1, predictions_test, lossfunc)

    # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = range(M)
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),4))
    z={}
    
    uhat_j_test={}
    
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        out_js = (fit_funct().fit(np.delete(x_train,j,1),y_train)).predict(np.delete(np.vstack((x_val,X1)),j,1)) 
        
        out_j  = out_js[:len(y_val)]
        uhat_j_test[idd] = out_js[len(y_val):]
    
        resids_drop = loss_func(y_val, out_j, lossfunc)
        z[idd] = resids_drop - resids_split
        inf_z[idd] = ztest(z[idd],alpha,bonf_correct =bonf)

    uhat_test=predictions_test ## TEST ERROR WITH J 
    target,err1,err2 = {},{},{}

    for idd,j in enumerate(ff): # ff include feature of interest
        err1[idd] = loss_func(Y1, uhat_j_test[idd], lossfunc)
        err2[idd] = loss_func(Y1, uhat_test, lossfunc)
        target[idd] = np.mean(err1[idd]-err2[idd])

    ### Result
    res= {}
    res['err2'] = err2
    res['inf']=inf_z
    res['z']=z
    res['target']=target
    res['resids_LOO']=resids_split
    res['resids_LOCO']=resids_drop
    res['err1'] = err1
   
    return res
