import pandas as pd 
import numpy as np 
import scipy
# import xlrd 
import sklearn

import numpy.linalg as linalg
import scipy.stats as stats
from scipy.stats import *
# from Gibbs_model_probit import Gibbs_sampling

# from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
# from utils import baseline_lr,baseline_esnet,baseline_justmean
# from utils import baseline_LogitElsnet,baseline_justmode,baseline_random,baseline_LogitLR,baseline_RanForest
# from sklearn.model_selection import KFold
from scipy.stats import binom 
from scipy.stats import norm
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestClassifier
from tqdm import trange
import time


def get_mean_var(X):
    dim,n = X.shape
    mean = np.sum(X,1)/n
    v = X - np.tile(mean.reshape([dim,1]), (1,n))
    v = np.sum(v**2,1)/(n-1)
    return np.hstack([mean.reshape([dim,1]), v.reshape([dim,1])])

#note: init with all 0 is very very important!!!
def GibbsSampling3(X, y, rho0, tau0,  max_iter, burn_in, r0 =1e-6):
    #augmented variable for probit likelihood
    t = 0.0*y
    #t = y #really bad
    N,dim = X.shape
    w = np.zeros(dim)
    #w = norm.rvs(scale=np.sqrt(tau0), size=dim)
    #w = linalg.solve(np.dot(X.transpose(), X) + 1e-6*np.diag(np.ones(dim)), np.dot(X.transpose(),y))
    lower = -np.inf*np.ones(N)
    upper = np.zeros(N)
    lower[y>0] = 0
    upper[y>0] = np.inf
    model = np.ndarray(6, dtype=np.dtype('object'))
    model[0] = np.zeros([dim, max_iter])
    model[1] = np.zeros([dim, max_iter])
    model[2] = np.zeros([N, max_iter])

    z = np.ones(dim)
    for iter in trange(max_iter):
        #sample t
        mu = np.dot(X,w)
        for n in range(N):
            t[n] = truncnorm.rvs(lower[n] - mu[n], upper[n] - mu[n], loc= mu[n])

        #sample w
        r = 1.0/tau0*np.ones(dim)
        r[z==0] = 1.0/r0

        V_inv = np.diag(r)+ np.dot(X.T, X)
        # V = np.linalg.inv()

        normal_sample = multivariate_normal(np.zeros(len(r)),np.ones(len(r)),allow_singular=True).rvs()
        cholek_upper = np.linalg.cholesky(V_inv).T
        # mu = np.dot(V, np.dot(X.T, t))
        mu =  np.linalg.solve(V_inv,np.dot(X.T, t))
        noise_part = np.linalg.solve(cholek_upper,normal_sample).squeeze()
        w = mu.squeeze() + noise_part.squeeze()
        # print('zhe:W_max:%.4f,W_min:%.4f'%(w.max(),w.min()))
        # print('zhe: mu_max:%.4f,mu_min:%.4f'%(mu.max(),mu.min()))
        # print('zhe:noise_part _max:%.4f,noise_part_min:%.4f's%(noise_part.max(),noise_part.min()))

        # w = multivariate_normal.rvs(mean=mu, cov=V) # svd not converge sometime, use Chlok instead

        #sample z
        pdf_ratio = norm.pdf(w, scale = np.sqrt(r0))/norm.pdf(w, scale=np.sqrt(tau0))
        theta = 1.0/(1.0+ (1.0 - rho0)/rho0 * pdf_ratio)
        z = binom.rvs(1, theta)
        
        # print ('iter %d'%iter)
        model[0][:,iter] = z.copy()
        model[1][:,iter] = w.copy()
        model[2][:,iter] = t.copy()

    model[3] = get_mean_var(model[0][:,-burn_in:])
    model[4] = get_mean_var(model[1][:, -burn_in:])
    model[5] = get_mean_var(model[2][:, -burn_in:])

    return model

if __name__ == '__main__':

    np.random.seed(0)
    data = np.load('simu-try-10.npy',allow_pickle=True,encoding='latin1')
    #X = data[0][:,[0,2]]
    #y = data[1]
    X = data[0]
    y = data[1]
    model = GibbsSampling3(X, y, 0.5, 100.0, 15000, 5000)
    #model = GibbsSampling_probit(X, y, 10.0, 15000, 5000)
    np.save('gibbs.simu-try-10.npy', model)