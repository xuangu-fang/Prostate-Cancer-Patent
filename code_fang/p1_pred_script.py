import pandas as pd
# from code_fang.utils_new import logis_lasso_regression_5_new 
import numpy as np 
from utils import logis_regression_5,kernel_svm_regression_5
from utils_new import feature_filter,all_model
from utils_new import logis_regression_5_new,logis_lasso_regression_5_new,gpr_regression_5_new,kernel_svm_regression_5_new

# to generate result sheet autoly
import pandas as pd 
import numpy as np 
import scipy
# import xlrd 
import sklearn

from utils import logis_regression_5,kernel_svm_regression_5,gpr_regression_5,logis_lasso_regression_5
from sklearn.model_selection import KFold

# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
from sklearn.linear_model import LogisticRegression
import pathlib


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# data_loading 
# data_table = pd.read_csv('../data/processed/all_feature_p1_lip_total_single.csv')
data_table_cont = pd.read_csv('../data/processed/all_feature_p1_spicie_cont_single.csv')
data_table_binary = pd.read_csv('../data/processed/all_feature_p1_spicie_binary_single.csv')
data_table_binary_standard = pd.read_csv('../data/processed/all_feature_p1_spicie_binary_standard_single.csv')

# last seven columns are labels
all_feature_cont = data_table_cont.iloc[:,1:-7]
all_feature_binary = data_table_binary.iloc[:,1:-7]
all_feature_binary_standard = data_table_binary_standard.iloc[:,1:-7]
# all_feature_binary

# for P1 patients, we use bin-split binarization process data
data_table = data_table_binary
# data_table = data_table_binary_standard


feature_dict = feature_filter(data_table)
one_feature_P1 = [['TG'],['Cer'],['DG']]
one_feature_P2 = [['Cer'],['Acy'],['Sph']]

two_feature_P1 = [['TG','Cer'],['TG','DG'],['DG','Cer']]
two_feature_P2 = [['Acy','Cer'],['Sph','Cer'],['Sph','Acy']]

three_feature_P1 = [['TG','Cer','DG']]
three_feature_P2 = [['Cer','Acy','Sph']]

index = ['logistic reg','logistic reg-lasso','linear-kernel svm','RBF-kernel svm','RBF-kernel GPR','matern-kernel GPR' ]



# set mertric list
mertric_list = ['accuracy','auc', 'precision','recall','specificity','NPV']
# mertric_list = ['accuracy']


# set rand-seed list
seed_list = [123,321,132,231,123]
# seed_list = [123]



# targets of P1
target ='1= death; 0=alive' 
# target = 'ADT_if_fail'

# targets of P2
# target ='gap_surv_time_class'
# sub_target = '_early_death'
# sub_target = '_long_survive'



Y = data_table_cont[target].values

if target == 'gap_surv_time_class':
    # for this target, Y=0 means early death, Y=1 means long survival, Y=2 for the middle value
    if sub_target == '_early_death':
        Y = (Y==0)*1 # early death predict
        target = target + sub_target
    elif sub_target == '_long_survive':
        Y = (Y==1)*1 # long survive predict
        target = target + sub_target

elif target == 'ADT_if_fail':
    # for this target, Y=0 means ADT success(not fail), otherwise ADT failure
    Y = (Y==0)*1

# single test gene:


kernel_white = DotProduct() + WhiteKernel()
kernel_RBF = RBF()
kernel_matern = Matern()

N_rand = len(seed_list)

for metric_name in mertric_list:
    

    result_table_mean = pd.DataFrame(index = index)
    result_table_std = pd.DataFrame(index = index)

    result_table_list = [pd.DataFrame(index = index) for i in range(N_rand)]

    for rand_seed,result_table in zip(seed_list,result_table_list):
        
        # np.random.seed(rand_seed)

        # run different features-setting along all models with current rand-seed
        X = data_table[feature_dict['gene']].values

        log_loss = logis_regression_5_new(X,Y,metric_name,rand_seed)
        log_lasso_loss = logis_lasso_regression_5_new(X,Y,0.1,metric_name,rand_seed)
        svm_loss_rbf = kernel_svm_regression_5_new(X,Y,'rbf',metric_name,rand_seed)
        svm_loss_linear = kernel_svm_regression_5_new(X,Y,'linear',metric_name,rand_seed)

        gpr_RBF_loss = gpr_regression_5_new(X,Y,kernel_RBF,metric_name,rand_seed)
        gpr_matern_loss = gpr_regression_5_new(X,Y,kernel_matern,metric_name,rand_seed)

        loss = [log_loss,log_lasso_loss,svm_loss_linear,svm_loss_rbf,gpr_RBF_loss,gpr_matern_loss]

        result_table['gene'] = loss


        all_model(data_table=data_table,\
                result_table=result_table,\
                features=one_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=False,metric_name=metric_name,rand_seed=rand_seed)

        all_model(data_table=data_table,\
                result_table=result_table,\
                features=one_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=True,metric_name=metric_name,rand_seed=rand_seed)

        all_model(data_table=data_table,\
                result_table=result_table,\
                features=two_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=False,metric_name=metric_name,rand_seed=rand_seed)

        all_model(data_table=data_table,\
                result_table=result_table,\
                features=two_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=True,metric_name=metric_name,rand_seed=rand_seed)   
                
        all_model(data_table=data_table,\
                result_table=result_table,\
                features=three_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=False,metric_name=metric_name,rand_seed=rand_seed)

        all_model(data_table=data_table,\
                result_table=result_table,\
                features=three_feature_P1,\
                feature_dict=feature_dict,Y=Y,\
                add_gene=True,metric_name=metric_name,rand_seed=rand_seed)

        print(result_table)


        path_str = './result_log_new/P1/' + target + '/' + metric_name 
        pathlib.Path(path_str).mkdir(parents=True, exist_ok=True) 
        file_name = path_str+'/P1_pred_'+target+ '_'+ metric_name+ str(rand_seed)+ '.csv'
        result_table.to_csv(file_name)
    

