import pandas as pd 
import numpy as np 
import scipy
# import xlrd 
import sklearn
from scipy.stats import multivariate_normal
from scipy.stats import binom 
from scipy.stats import norm

from tqdm import trange
import time
from utils import my_sigmoid

# Gibbs sampling class

# no parallel first, to be added in the future

class Gibbs_sampling:
    def __init__(self,data_dict ,init_paras, hyper_paras):

        self.X = data_dict['X_tr']
        self.Y = data_dict['y_tr']

        self.N_sample, self.N_feature = self.X.shape 
        self.X_test = data_dict['X_test']
        self.Y_test = data_dict['y_test']

        self.INTERVAL = hyper_paras['INTERVAL']
        self.BURNING = hyper_paras['BURNING']
        self.MAX_NUMBER = hyper_paras['MAX_NUMBER']
        self.VALITA_INTERVAL = hyper_paras['VALITA_INTERVAL']
        self.alpha = hyper_paras['alpha']
        self.beta = hyper_paras['beta']
        self.r0 = hyper_paras['r0']
        self.r1 = hyper_paras['r1']
        self.JITTER = hyper_paras['JITTER'] # use logpdf

        self.z = init_paras['z'].copy()
        self.s = init_paras['s'].copy()
        self.b = init_paras['b']
        self.tau = init_paras['tau']
        self.W = init_paras['W'].copy()

        self.z_mean = self.z.copy()
        self.s_mean = self.z.copy()
        self.W_mean = np.zeros(self.N_feature)

        self.z_collect = []
        self.s_collect = []
        self.b_collect = []
        self.tau_collect = []
        self.W_collect = []

        self.a_0 = init_paras['a0']
        self.b_0 = init_paras['b0']

        self.a_hat = init_paras['a0']
        self.b_hat = init_paras['b0']

        self.gauss1 = norm(0,np.sqrt(self.r1))
        self.gauss2 = norm(0,np.sqrt(self.r0))

    def s_sample(self):
        for i,z_k in enumerate(self.z):
            if z_k > 0: #z_k=1
                v_k = my_sigmoid(np.log(self.beta/(1-self.beta)) + \
                    self.gauss1.logpdf(self.W[i]) \
                    - self.gauss2.logpdf(self.W[i]))
                # print(v_k)
                self.s[i] = np.random.binomial(size=len(self.s[i]), n=1, p= v_k)
            else: #z_k=0
                self.s[i] = np.random.binomial(size=len(self.s[i]), n=1, p= self.beta)

    def z_sample(self):
        for i in range(len(self.z)):
            sum_term = (self.gauss1.logpdf(self.W[i]) 
            - self.gauss2.logpdf(self.W[i]) ) * self.s[i] # mask out items whose s_kj=0
            eta_k = my_sigmoid(np.log(self.beta/(1-self.beta)) + np.sum(sum_term))
            self.z[i] = np.random.binomial( n=1, p= eta_k)

    def tau_sample(self):
        concat_W = np.zeros(self.N_feature)
        concat_W[:-1] = np.concatenate((self.W))
        concat_W[-1] = self.b
        err = self.Y.reshape(-1,1) - np.matmul(self.X, concat_W).reshape(-1,1)

        self.a_hat = self.a_0 + 0.5* self.N_sample
        self.b_hat = self.b_0 + 0.5* np.matmul(np.transpose(err),err)[0][0]

        self.tau = np.random.gamma(shape=self.a_hat, scale=1.0/self.b_hat, size=None)

    def w_sample(self):
        R = np.ones(self.N_feature)
        zs =  np.concatenate([self.z[i] * self.s[i] for i in range(len(self.z)) ]) # z_k * s_kj
        R[:-1] = np.where(zs > 0, self.r1, self.r0)
        R[-1] = self.r1 

        sigma_inv = np.diag(1.0/R) \
            + self.tau * np.matmul(np.transpose(self.X),self.X)
        # sigma = np.linalg.inv(sigma_inv ) + np.diag(0.01*np.ones(len(R)))# size: (N_feature+1) * (N_feature+1)

        # where_are_NaNs = np.isnan(sigma)
        # sigma[where_are_NaNs] = 0.0

        # where_are_infs = np.isinf(sigma)
        # sigma[where_are_infs] = 100.0

        # print('is nan',np.isnan(np.sum(sigma)))
        # print('is inf',np.isinf(np.sum(sigma)))

        mu = np.linalg.solve(sigma_inv, \
            self.tau*np.matmul(np.transpose(self.X),self.Y.reshape(-1,1))).squeeze()


        # svd not converge bug while diretly apply multivariate_normal, use cholek instead
        # dis = multivariate_normal(mu,sigma,allow_singular=True)
        # W_new = dis.rvs()
        # W_new = np.random.multivariate_normal(mu,sigma) # check if can use prec mat:no

        normal_sample = multivariate_normal(np.zeros(len(R)),np.ones(len(R)),allow_singular=True).rvs()
        cholek_upper = np.linalg.cholesky(sigma_inv).T
        noise_part = np.linalg.solve(cholek_upper,normal_sample).squeeze()
        W_new = mu + noise_part

        self.b = W_new[-1]
        offset = 0
        for i in range(len(self.W)):
            group_len = len(self.W[i])
            self.W[i] = W_new[offset:offset+group_len]
            offset = offset + group_len
        
    def sample_collect(self):
        self.z_collect.append(self.z)
        self.s_collect.append(self.s)
        self.b_collect.append(self.b)
        self.tau_collect.append(self.tau)
        self.W_collect.append(self.W)    
    
    def model_run(self):
        counts = 0
        for i in trange(self.MAX_NUMBER):
            self.z_sample()
            self.s_sample()
            self.w_sample()
            self.tau_sample()
            if i>=self.BURNING and i%self.INTERVAL==0:
                self.sample_collect()
            if i%self.VALITA_INTERVAL==0:
                self.model_run_test()
        self.model_test()

    def model_test(self): # avg on the indicator
        N_collect = len(self.W_collect)
        if N_collect > 0:
            W_samples = np.zeros((N_collect,self.N_feature))
            for i in range(N_collect):
                W_samples[i,:-1] = np.concatenate((self.W_collect[i]))
                W_samples[i,-1] = self.b_collect[i]

            W_avg = np.mean(W_samples,axis=0).reshape(-1,1)


            # compute the avg indicators
            s_mean = []
            for k_id in range(len(self.z)):
                s_sample = np.stack([self.s_collect[i][k_id] for i in range(len(self.s_collect))])
                s_mean_sub = np.mean(s_sample,0)
                s_mean_sub = np.where(s_mean_sub>0.9,1,0)
                s_mean.append(s_mean_sub)

            z_sample = np.stack(self.z_collect)
            z_mean = np.mean(z_sample,0)
            z_mean = np.where(z_mean>0.9,1,0)

            indicators = np.zeros((self.N_feature))
            indicators[:-1] = np.concatenate([s_mean[i] * z_mean[i] for i in range(len(z_mean))])
            indicators[-1] = 1.0

            sparse_W = indicators.reshape(-1,1) * W_avg

            predict = np.matmul(self.X_test, sparse_W).squeeze()
            rmse = np.sqrt(np.mean((predict-self.Y_test.squeeze())**2))
            print("\n\n final test rmse = %.5f"%(rmse))

            self.z_mean = z_mean
            self.s_mean = s_mean
            self.W_mean = sparse_W.squeeze() 
            return rmse

        else: print('no collect samples yet')

    def model_run_test(self):
            concat_W = np.zeros(self.N_feature)
            concat_W[:-1] = np.concatenate((self.W))
            concat_W[-1] = self.b
            predict_test = np.matmul(self.X_test, concat_W.reshape(-1,1)).squeeze()
            predict_train = np.matmul(self.X, concat_W.reshape(-1,1)).squeeze()
            print("\n running test-rmse = %.5f"%(np.sqrt(np.mean((predict_test-self.Y_test.squeeze())**2))))
            print("running train-rmse = %.5f\n"%(np.sqrt(np.mean((predict_train-self.Y.squeeze())**2))))
