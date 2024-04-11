'''
new util-funcs on feature split
'''


import pandas as pd 
import numpy as np 
import sklearn
import scipy
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
# from sklearn.metrics import 
from sklearn.metrics import confusion_matrix 
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF,Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC




def metric_score(y_pred,y_pred_prob,y_test,metric_name='accuracy'):
    

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    if metric_name=='accuracy':
        return accuracy_score(y_test, y_pred)
    elif metric_name=='auc':
        return roc_auc_score(y_test, y_pred_prob[:,1])
    elif metric_name=='precision':
        return precision_score(y_test, y_pred)
    elif metric_name=='recall':
        return recall_score(y_test, y_pred)
    elif metric_name=='specificity':
        return float(tn)/(tn + fp)
    elif metric_name=='NPV':
        return float(tn)/(tn + fn)

def logis_regression_5_new(X,Y,metric_name='accuracy',rand_seed=12):
    
    kf = KFold(n_splits=5,random_state=rand_seed,shuffle=True)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = LogisticRegression(random_state=rand_seed).fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        test_score = metric_score(y_pred,y_pred_prob,y_test,metric_name)
        
        loss.append(test_score)
    
    return np.array(loss).mean()


def logis_lasso_regression_5_new(X,Y,l1_ratio=0.1,metric_name='accuracy',rand_seed=123):
    
    kf = KFold(n_splits=5,random_state=rand_seed,shuffle=True)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = LogisticRegression(random_state=rand_seed,penalty='elasticnet',solver='saga',l1_ratio=l1_ratio
).fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        test_score = metric_score(y_pred,y_pred_prob,y_test,metric_name)
        
        loss.append(test_score)
    
    return np.array(loss).mean()


def gpr_regression_5_new(X,Y,kernel=DotProduct() + WhiteKernel(),metric_name='accuracy',rand_seed=123):
    
    kf = KFold(n_splits=5,random_state=rand_seed,shuffle=True)
    kf.get_n_splits(X)

    loss = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


        clf = GaussianProcessClassifier(kernel=kernel,random_state=rand_seed)

        clf.fit(X_train, y_train)
        # test_score = clf.score(X_test,y_test)
        # pred_y = np.where(clf.predict(X_test)>0.5,1,0)
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        test_score = metric_score(y_pred,y_pred_prob,y_test,metric_name)
        loss.append(test_score)

    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()

def kernel_svm_regression_5_new(X,Y,kernel='rbf',metric_name='accuracy',rand_seed=123):
    
    kf = KFold(n_splits=5,random_state=rand_seed,shuffle=True)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel=kernel, probability=True))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        test_score = metric_score(y_pred,y_pred_prob,y_test,metric_name)
        # print(test_score)
        loss.append(test_score)
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()


# P1(mHSPC) testtosterone suppression failure: CER(ceramide), DG(diacylglycerol), TG(triaclyglycerol)
# P2(mCRPC) overall survival:  CER(ceramide), Sph(Sphingosine), Acylcarnitine(acylcarnitine)
def feature_filter(df):
    all_feature = list(df)
    feature_dict = {}
    feature_dict['gene'] = [item for item in all_feature if 'gene_' in item]
    # feature_dict['TG'] = [item for item in all_feature if 'lip_P1' in item and 'lip_P1_TG' not in item] # remove the TG lip
    # lip_feature_P2 = [item for item in all_feature if 'lip_P2' in item and '_Cer' not in item]
    # lip_feature_all =  [item for item in all_feature if 'lip_' in item and 'P1_Cer' not in item]
    

    feature_dict['TG'] = [item for item in all_feature if 'lip_P1_TG' in item]
    # feature_dict['Cer'] = [item for item in all_feature if 'lip_P1_Cer' in item]
    feature_dict['Cer'] = [item for item in all_feature if '_Cer' in item]

    feature_dict['DG'] = [item for item in all_feature if 'lip_P1_DG' in item]


    feature_dict['Acy'] = [item for item in all_feature if '_Acylcarnitine' in item]
    feature_dict['Sph'] = [item for item in all_feature if '_Sph' in item]
    feature_dict['sig'] = [item for item in all_feature if 'signature' in item]

    return feature_dict

def all_model(data_table,result_table,features,feature_dict,Y,add_gene=False,metric_name='accuracy',rand_seed=123):
    
    kernel_RBF = RBF()
    kernel_matern = Matern()
    for fes in features: 
        
        feature_name = '_'.join(fes)
        feature_list = [data_table[feature_dict[item]].values for item in fes]

        if add_gene:
            feature_name = 'gene_'+feature_name
            feature_list.append(data_table[feature_dict['gene']].values)

        X = np.concatenate(feature_list,1)
        log_loss = logis_regression_5_new(X,Y,metric_name,rand_seed)
        log_lasso_loss = logis_lasso_regression_5_new(X,Y,0.1,metric_name,rand_seed)
        svm_loss_rbf = kernel_svm_regression_5_new(X,Y,'rbf',metric_name,rand_seed)
        svm_loss_linear = kernel_svm_regression_5_new(X,Y,'linear',metric_name,rand_seed)

        gpr_RBF_loss = gpr_regression_5_new(X,Y,kernel_RBF,metric_name,rand_seed)
        gpr_matern_loss = gpr_regression_5_new(X,Y,kernel_matern,metric_name,rand_seed)

        loss = [log_loss,log_lasso_loss,svm_loss_linear,svm_loss_rbf,gpr_RBF_loss,gpr_matern_loss]
        
        result_table[feature_name] = loss

