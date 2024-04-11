import pandas as pd 
import numpy as np 
import scipy
# import xlrd 
import sklearn
import sys
from Gibbs_model_probit import Gibbs_sampling

from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from utils import baseline_lr,baseline_esnet,baseline_justmean
from utils import baseline_LogitElsnet,baseline_justmode,baseline_random,baseline_LogitLR,baseline_RanForest,baseline_Gibbs_zhe,baseline_LogitElsnetCV
from sklearn.model_selection import KFold
from scipy.stats import binom 
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange
import time

# data_loading 
np.random.seed(1213)
data_table = pd.read_csv('../data/processed/all_feature_p1_lip_specie.csv')
target = '1= death; 0=alive'


# normalization

# min-max
df = data_table[target]


# check nan
data_table[target].isnull().values.any()
data_table.fillna(data_table.mean(), inplace=True) # fill nan with column mean

all_feature = data_table.iloc[:,1:-7]
#lip_feature = data_table.iloc[:,1:775]#for p2
# gene_feature = data_table.iloc[:,1:-372]#for p1

Y = data_table[target].values

feature_names = list(all_feature.columns)
K_lip=41 # group number, from data process notebook
K_gene=3
K = K_lip + K_gene 

group_ind_dict = {}
group_ind = []
group_ind_concat = []
for i in range(K_lip):
    group_ind_dict['lip'+'_'+str(i)] = []
for i in range(K_gene):
    group_ind_dict['gene'+'_'+str(i)] = []

for name in feature_names:
    cate = name.split('_')[0]
    id = name.split('_')[-1]
    group_ind_dict[cate+'_'+id].append(name)

for i in range(K_lip):
    group_ind.append(group_ind_dict['lip'+'_'+str(i)])
    group_ind_concat = group_ind_concat + group_ind_dict['lip'+'_'+str(i)]

for i in range(K_gene):
    group_ind.append(group_ind_dict['gene'+'_'+str(i)])
    group_ind_concat = group_ind_concat + group_ind_dict['gene'+'_'+str(i)]

# group_ind

# re-arrange the features of X based on the group split order
X_new = all_feature[group_ind_concat].values

N_sample, _ = X_new.shape
# add all-one column at the last 
bias_col = np.ones(N_sample).reshape((N_sample,1))
X_new = np.concatenate((X_new,bias_col),axis=1)

print(X_new.shape)

# init hyper-parameters
alpha = 0.5
beta = 0.60
r0 = 1e-6
r1 = 100.0
a0 = 1.0
b0 = 1.0
JITTER = 1e-3

INTERVAL = 1
VALITA_INTERVAL = 100
BURNING = 600 
MAX_NUMBER = 1000

hyper_paras = {'INTERVAL':INTERVAL, 'BURNING':BURNING,'MAX_NUMBER':MAX_NUMBER,'VALITA_INTERVAL':VALITA_INTERVAL,
'alpha':alpha, 'beta':beta,'r0':r0,'r1':r1,'JITTER':JITTER}

# init parameters with lr_result
def get_init_paras(w_lr):
    z_array_init = np.ones(K) #np.random.binomial(size=K, n=1, p= alpha)
    s_list_init = [np.ones(len(item)) for item in group_ind]
    # s_list_init = [np.ones(len(item)) for item in group_ind]#[np.random.binomial(size=len(item), n=1, p= beta) for item in group_ind]
    b_init = w_lr[-1]#np.random.normal(loc=0.0, scale=r1,size=None)
    # tau_init = 1.0#np.random.gamma(shape=alpha, scale=1.0/beta, size=None)

    W_init = []
    offset=0
    for i in range(K):
        # mask1 = 1-z_array_init[i] * s_list_init[i]
        # mask2 = z_array_init[i] * s_list_init[i]
        # spike = np.random.normal(loc=0.0, scale=r0,size=len(s_list_init[i]))
        # slab = np.random.normal(loc=0.0, scale=r1,size=len(s_list_init[i]))
        # W_group = spike * mask1 + slab * mask2

        
        group_len = len(s_list_init[i])
        W_group= w_lr[offset:offset+group_len]
        offset = offset + group_len
        W_init.append(W_group)

    init_paras = {'z':z_array_init, 's':s_list_init, 'b':b_init,  'W':W_init,'a0':a0,'b0':b0}
    return init_paras

def zero_init_paras():
    z_array_init = np.ones(K) #np.random.binomial(size=K, n=1, p= alpha)
    s_list_init = [np.ones(len(item)) for item in group_ind]
    # s_list_init = [np.ones(len(item)) for item in group_ind]#[np.random.binomial(size=len(item), n=1, p= beta) for item in group_ind]
    b_init = 0.0#np.random.normal(loc=0.0, scale=r1,size=None)
    # tau_init = 1.0#np.random.gamma(shape=alpha, scale=1.0/beta, size=None)

    W_init = [np.zeros(len(item)) for item in group_ind]

    init_paras = {'z':z_array_init, 's':s_list_init, 'b':b_init,  'W':W_init,'a0':a0,'b0':b0}
    return init_paras
    
N = 25
start = time.time()
lr_acc = np.zeros(N)
rf_acc = np.zeros(N)
esnet_acc = np.zeros(N)
mode_acc = np.zeros(N)
random_acc = np.zeros(N)
ours_acc = np.zeros(N)
zhe_gibs_acc = np.zeros(N)

lr_auc = np.zeros(N)
rf_auc = np.zeros(N)
esnet_auc = np.zeros(N)
ours_auc = np.zeros(N)
zhe_gibs_auc = np.zeros(N)

for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y.squeeze(),test_size=0.3)

    data_dict = {'X_tr':X_train, 'y_tr':y_train, 'X_test':X_test, 'y_test':y_test}  
    dict_lr = baseline_LogitLR(data_dict)
    dict_els = baseline_LogitElsnetCV(data_dict)
    dict_rf = baseline_RanForest(data_dict)
    dict_mode = baseline_justmode(data_dict)
    dict_random = baseline_random(data_dict)
    dict_gibbs_zhe = baseline_Gibbs_zhe(data_dict,hyper_paras)

    model = Gibbs_sampling(data_dict,get_init_paras(dict_lr['clf'].coef_.squeeze()), hyper_paras)
    # model = Gibbs_sampling(data_dict,zero_init_paras(), hyper_paras)
    dict_ours = model.model_run()

    lr_acc[i] = dict_lr['acr']
    esnet_acc[i] = dict_els['acr']
    rf_acc[i] = dict_rf['acr']
    mode_acc[i] = dict_mode['acr']
    random_acc[i] = dict_random['acr']
    ours_acc[i] = dict_ours['acr']
    zhe_gibs_acc[i] = dict_gibbs_zhe['acr']

    lr_auc[i] = dict_lr['auc']
    rf_auc[i] = dict_rf['auc']
    esnet_auc[i] = dict_els['auc']
    ours_auc[i] = dict_ours['auc']
    zhe_gibs_auc[i] = dict_gibbs_zhe['auc']

    print('avg_W_zhe max:%.3f,avg_W_zhe min:%.3f'%(dict_gibbs_zhe['model'][4][:,0].max(),dict_gibbs_zhe['model'][4][:,0].min()))
    print('avg_W_ours max:%.3f,avg_W_zhe min:%.3f'%(model.W_mean.max(),model.W_mean.min()))

print('\n\nours_acr_mean: %.4f,ours_acr_std: %.4f '%( ours_acc.mean(), ours_acc.std() ) )
print('gibbs_zhe_acr_mean: %.4f,gibbs_zhe_acr_std: %.4f '%(zhe_gibs_acc.mean(),zhe_gibs_acc.std() ) )
print('lr_acr_mean: %.4f,lr_acr_std: %.4f '%(lr_acc.mean(),lr_acc.std() ) )
print('esnet_acr_mean: %.4f,esnet_acr_std: %.4f '%(esnet_acc.mean(),esnet_acc.std() ) )
print('rf_acr_mean: %.4f,rf_acr_mean: %.4f '%(rf_acc.mean(),rf_acc.std() ) )
print('just-mode_acr_mean: %.4f,mode_acr_std: %.4f '%(mode_acc.mean(),mode_acc.std() ) )
print('just-random_acr_mean: %.4f,just-random_acr_std: %.4f '%(random_acc.mean(),random_acc.std() ) )


print('\nours_AUC_mean: %.4f,ours_AUC_std: %.4f '%(ours_auc.mean(),ours_auc.std() ) )
print('gibbs_zhe_AUC_mean: %.4f,gibbs_zhe_AUC_std: %.4f '%(zhe_gibs_auc.mean(),zhe_gibs_auc.std() ) )
print('lr_AUC_mean: %.4f,lr_AUC_std: %.4f '%(lr_auc.mean(),lr_auc.std() ) )
print('esnet_AUC_mean: %.4f,esnet_AUC_mean: %.4f '%(esnet_auc.mean(),esnet_auc.std() ) )
print('rf_AUC_mean: %.4f,rf_AUC_std: %.4f '%(rf_auc.mean(),rf_auc.std() ) )
print('setting: lr-init, update z')


f= open("result_log/P1_live_dead_all_feature.txt","a+")
f.write('\n take %g seconds to finish '%(time.time()-start))
f.write('\nsetting: lr-init, update z')
f.write('\n Setting: alpha = %f, beta =%f, BURNING = %d, MAX_NUMBER = %d ,N = %d, alpha=%.3f, beta=%.3f\n\n'%(alpha,beta,BURNING,MAX_NUMBER,N,alpha,beta))

f.write('\n\nours_acr_mean: %.4f,ours_acr_std: %.4f '%( ours_acc.mean(), ours_acc.std() ) )
f.write('\ngibbs_zhe_acr_mean: %.4f,gibbs_zhe_acr_std: %.4f '%(zhe_gibs_acc.mean(),zhe_gibs_acc.std() ) )
f.write('\nlr_acr_mean: %.4f,lr_acr_std: %.4f '%(lr_acc.mean(),lr_acc.std() ) )
f.write('\nesnet_acr_mean: %.4f,esnet_acr_std: %.4f '%(esnet_acc.mean(),esnet_acc.std() ) )
f.write('\nrf_acr_mean: %.4f,rf_acr_mean: %.4f '%(rf_acc.mean(),rf_acc.std() ) )
f.write('\njust-mode_acr_mean: %.4f,mode_acr_std: %.4f '%(mode_acc.mean(),mode_acc.std() ) )
f.write('\njust-random_acr_mean: %.4f,just-random_acr_std: %.4f '%(random_acc.mean(),random_acc.std() ) )


f.write('\n\nours_AUC_mean: %.4f,ours_AUC_std: %.4f '%(ours_auc.mean(),ours_auc.std() ) )
f.write('\ngibbs_zhe_AUC_mean: %.4f,gibbs_zhe_AUC_std: %.4f '%(zhe_gibs_auc.mean(),zhe_gibs_auc.std() ) )
f.write('\nlr_AUC_mean: %.4f,lr_AUC_std: %.4f '%(lr_auc.mean(),lr_auc.std() ) )
f.write('\nesnet_AUC_mean: %.4f,esnet_AUC_mean: %.4f '%(esnet_auc.mean(),esnet_auc.std() ) )
f.write('\nrf_AUC_mean: %.4f,rf_AUC_std: %.4f \n'%(rf_auc.mean(),rf_auc.std() ) )

f.write('\n\n\n')
f.close()