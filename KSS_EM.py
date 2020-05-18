import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

import csv
import addfips as af

############################
# KSS
############################

def compute_KSS_statistic(inds,C,B):
    c = np.sum(C[inds])
    b = np.sum(B[inds])
    
    call = np.sum(C)
    ball = np.sum(B)
    
    if c > 0 and call > 0 and (c/b) > (call/ball):
        return c*np.log(c/b) + (call-c)*np.log((call-c)/(ball-b)) - call*np.log(call/ball)
    else:
        return 0

def KSS(C,B):
    div = C / B
    sorted_inds = np.argsort(div)[::-1]

    cur_max = -np.Inf
    for i in range(len(sorted_inds)):
        inds = sorted_inds[:i+1]
        kss = compute_KSS_statistic(inds,C,B)
        if kss < cur_max:
            break
        else:
            cur_inds = inds
            cur_max = kss
    return cur_inds

############################
# EM
############################

def compute_responsibilities(C,B,qin,qout,alpha):
    num = alpha*poisson.pmf(C,qin*B)
    denom = num + (1-alpha)*poisson.pmf(C,qout*B)
    
    return num/denom

def compute_log_lik(C,B,qin,qout,alpha):
    x = alpha*poisson.pmf(C,qin*B)
    y = (1-alpha)*poisson.pmf(C,qout*B)
    return np.nansum(np.log(x+y))

def single_em(C,B,qin_init=1, qout_init=1, alpha_init=0.05, tol=1e-3, max_num_iter=1000, min_num_iter = 100):
    resps_init = compute_responsibilities(C,B,qin_init,qout_init, alpha_init)
    
    loglik_prev = -np.inf
    
    qin_cur = qin_init
    qout_cur = qout_init
    alpha_cur = alpha_init
    resps_cur = resps_init
    
    # make lists for recording params
    qin_list = np.zeros(max_num_iter+1)
    qout_list = np.zeros(max_num_iter+1)
    alpha_list = np.zeros(max_num_iter+1)
    
    qin_list[0] = qin_init
    qout_list[0] = qout_init
    alpha_list[0] = alpha_init
    
    for _ in range(max_num_iter):
        qin_cur = np.sum(resps_cur*C) / np.sum(resps_cur*B)
        qout_cur = np.sum((1-resps_cur)*C) / np.sum((1-resps_cur)*B)
        alpha_cur = np.mean(resps_cur)
        
        qin_list[_+1] = qin_cur
        qout_list[_+1] = qout_cur
        alpha_list[_+1] = alpha_cur
        
        # M step: update resps
        resps_cur = compute_responsibilities(C,B,qin_cur,qout_cur, alpha_cur)
        
        # check for convergence
        loglik_cur = compute_log_lik(C,B,qin_cur,qout_cur, alpha_cur)
        if _ > min_num_iter and np.abs((loglik_cur - loglik_prev) / loglik_prev) < tol:
            break
        else:
            loglik_prev = loglik_cur
    return (qin_list, qout_list, alpha_list, resps_cur, loglik_cur)

def em(C,B):
    alpha_init_list = np.arange(0,0.1,0.05)[1:]
    qin_init_list = np.arange(0.5,10,0.1)
    qout_init_list = np.arange(0.5,5,0.1)
    
    loglik_max = -np.Inf
    
    for alpha_init in alpha_init_list:
        for qin_init in qin_init_list:
            for qout_init in qout_init_list:
                if qin_init > qout_init:
                    qin_list, qout_list, alpha_list, resps, loglik_cur = single_em(C,B,qin_init=qin_init,qout_init=qout_init,alpha_init=alpha_init, max_num_iter=1000,min_num_iter=50)
                    if loglik_cur > loglik_max:
                        qin_list_max = qin_list
                        qout_list_max = qout_list
                        alpha_list_max = alpha_list
                        resps_max = resps

                    loglik_max = loglik_cur
    qin = qin_list_max[np.nonzero(qin_list_max)][-1]
    qout = qout_list_max[np.nonzero(qout_list_max)][-1]
    alpha = alpha_list_max[np.nonzero(alpha_list_max)][-1]
    
    # switch qin, qout
    if qin < qout:
        qin_new = qout
        qout_new = qin
        qin = qin_new
        qout = qout_new
        alpha = 1-alpha
        resps = 1-resps
    return qin,qout,alpha,resps_max

