import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from construct_moment import cumulant_estimate, multiplier_coefficients, multipliers

def all_together_cross_fitting(x, p, q, r, 
                 model_p=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]),
                 model_q=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100])):
    
    res_p = np.zeros(x.shape[0])
    res_q = np.zeros(x.shape[0])
    mult_p = np.zeros((r+1,x.shape[0]))
    mult_p_est = np.zeros((r+1,x.shape[0]))
    mult_p_est_split = np.zeros((r+1,x.shape[0]))
    moments_p_est = np.zeros(r+1)
    cum_list= []

    robust_ortho_est_ml = []
    robust_ortho_var = []
    
    kf = KFold(n_splits=2)
    for train_index, remain_index in kf.split(x):
        nested_kf = KFold(n_splits=2)
        x_train, p_train, q_train = x[train_index], p[train_index], q[train_index]
        x_remain = x[remain_index]
        
        for cum_index, test_index in kf.split(x_remain):
            # Split the data in half, train and test
            x_cum, p_cum  = x[remain_index[cum_index]], p[remain_index[cum_index]]
            x_test, p_test, q_test  = x[remain_index[test_index]], p[remain_index[test_index]], q[remain_index[test_index]]
            
            # Fit with LassoCV the treatment as a function of x and the outcome as
            # a function of x, using only the train fold
            model_p.fit(x_train, p_train)
            model_q.fit(x_train, q_train)
            
            # Then compute residuals p-g(x) and q-q(x) on test fold
            res_p[cum_index] = (p_cum - model_p.predict(x_cum)).flatten()
            

            res_p[test_index] = (p_test - model_p.predict(x_test)).flatten()
            res_q[test_index] = (q_test - model_q.predict(x_test)).flatten()
    
            cum_list += cumulant_estimate(r, res_p[cum_index])
            for i in range(1, r+1):
                cf_list = multiplier_coefficients(cum_list[0:i+1])
                
                multiplier_list = multipliers(i, cf_list, res_p[test_index])
                
                mult_p_est_split[i, test_index] = np.sum(multiplier_list, axis = 0)
                theta_est = np.mean(res_q[test_index] * multiplier_list)/np.mean(res_p[test_index] * multiplier_list)
                robust_ortho_est_ml.append(theta_est)
                robust_ortho_var.append(np.mean((res_q[test_index] - theta_est * res_p[test_index])**2 * multiplier_list**2))
                
        
        
    return robust_ortho_est_ml, robust_ortho_var, model_p.coef_, model_q.coef_, cum_list
