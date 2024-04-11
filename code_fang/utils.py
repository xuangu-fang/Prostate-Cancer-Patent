import pandas as pd 
import numpy as np 
import sklearn
import scipy
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split
from functools import reduce

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV,RidgeCV
from Gibbd_zhe import GibbsSampling3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def get_sparse_name(clf,var_name,thr=1e-4,l1_ratio=0.005,prefix = 'analysis result/p1_total_dead_live',save_txt = False):
    sparse = clf.coef_.squeeze()
    ind_list = np.where(np.abs(sparse)>thr)[0]
    select_set = []

    select_set_gene = []
    select_set_lip = []
    
    gene_feature_counts = 0
    lip_feature_counts = 0
    clin_feature_counts = 0

    for index in ind_list:
            
            select_set.append((var_name[index],'%.3f'%(sparse[index])))
            # select_set[var_name[index]] = sparse[index]

            if var_name[index][0] == 'g':
                gene_feature_counts = gene_feature_counts + 1
                select_set_gene.append((var_name[index],'%.3f'%(sparse[index])))

            elif  var_name[index][0] == 'l':
                lip_feature_counts = lip_feature_counts + 1
                select_set_lip.append((var_name[index],'%.3f'%(sparse[index])))
            else:
                clin_feature_counts = clin_feature_counts + 1
    
    print('with l1_ratio setting = %.4f, sparse ratio %.4f (%d/%d)'%(l1_ratio,len(ind_list)/len(sparse), len(ind_list),len(sparse)))
    print('selected gene features: %d, lip features: %d, clin features: %d'%(gene_feature_counts,lip_feature_counts,clin_feature_counts))


    select_set.sort(reverse = True,key=lambda x:abs(float(x[1])))
    select_set_gene.sort(reverse = True,key=lambda x:abs(float(x[1])))
    select_set_lip.sort(reverse = True,key=lambda x:abs(float(x[1])))

    if save_txt:

        f = open( prefix+' %.4f.txt'%(l1_ratio),"w+")
        f.write('with l1_ratio setting = %.4f, sparse ratio %.4f (%d/%d) \n'%(l1_ratio,len(ind_list)/len(sparse), len(ind_list),len(sparse)))
        f.write('selected gene features: %d, lip features: %d, clin features: %d \n'%(gene_feature_counts,lip_feature_counts,clin_feature_counts))


        for element in select_set:
            f.write(str(element))
            f.write('\n')
        f.close()

    return select_set,select_set_gene,select_set_lip

def multi_lasso(x,y,new_table,N=50,l1_ratio=0.005,C=0.8):
    # l1_ratio = 0.005


    # N=50
    acr_array = np.zeros((N,1))
    auc_array = np.zeros((N,1))
    fpr_array = np.zeros((N,1))

    select_feature_list = []
    select_feature_list_gene = []
    select_feature_list_lip = []


    for i in range(N):

        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                            stratify=y,
                                                            test_size=0.1
                                                            )


        clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',l1_ratio=l1_ratio,C=0.8).fit(X_train,y_train)
        # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        # print(y_pred)
        # print(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp/(fp+tn)
        acr = (y_pred == y_test).sum()/len(y_test)
        auc = roc_auc_score(y_test, y_pred_prob[:,1])

        acr_array[i] = acr
        auc_array[i] = auc
        fpr_array[i] = fpr
        print('accuracy is %f, auc is %f, fpr is %f'%(acr,auc,fpr))

        select_set,select_set_gene, select_set_lip = get_sparse_name(clf,list(new_table)[1:-3],thr=3e-2,l1_ratio=l1_ratio,prefix = 'p2_specie_live_dead',save_txt = False)
        
        select_feature_list.append([item[0] for item in select_set])
        select_feature_list_lip.append([item[0] for item in select_set_lip])
        select_feature_list_gene.append([item[0] for item in select_set_gene])


    print('avg_accuracy is %f, avg_auc is %f, avg_fpr is %f'%(acr_array.mean(),auc_array.mean(),fpr_array.mean()))
    print('std_accuracy is %f, std_auc is %f, std_fpr is %f'%(acr_array.std(),auc_array.std(),fpr_array.std()))

    return select_feature_list,select_feature_list_gene,select_feature_list_lip

def my_ridge(inter_select_feature,y,new_table,N=50,C=5,smooth_name = 'hard'):
    N=50
    acr_array = np.zeros((N,1))
    auc_array = np.zeros((N,1))
    fpr_array = np.zeros((N,1))

    new_x = new_table[inter_select_feature].values
    new_y = y

    select_feature_list = []

    for i in range(N):

        X_train, X_test, y_train, y_test = train_test_split(new_x, new_y,
                                                            stratify=new_y,
                                                            test_size=0.1
                                                            )


        clf = LogisticRegression(random_state=0,penalty='l2',C=C).fit(X_train,y_train)
        # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        # print(y_pred)
        # print(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp/(fp+tn)
        acr = (y_pred == y_test).sum()/len(y_test)
        auc = roc_auc_score(y_test, y_pred_prob[:,1])

        acr_array[i] = acr
        auc_array[i] = auc
        fpr_array[i] = fpr
        print('accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))

        # select_set,select_set_gene, select_set_lip = get_sparse_name(clf,list(new_table)[1:-5],thr=5e-2,prefix = 'p1_total_ADT_fail')
        
        # select_feature_list.append([item[0] for item in select_set])
    print('with smooth method:' + smooth_name)
    print('avg_accuracy is %f, avg_auc is %f, avg_fpr is %f'%(acr_array.mean(),auc_array.mean(),fpr_array.mean()))
    print('std_accuracy is %f, std_auc is %f, std_fpr is %f'%(acr_array.std(),auc_array.std(),fpr_array.std()))

    res_str = 'with smooth method:' + smooth_name + '\n' + 'avg_accuracy is %f, avg_auc is %f, avg_fpr is %f'%(acr_array.mean(),auc_array.mean(),fpr_array.mean())
    return clf,res_str

def hard_smooth(select_feature_list):
    # smooth_feature = list(reduce(set.intersection, [set(item) for item in select_feature_list]))
    return list(reduce(set.intersection, [set(item) for item in select_feature_list]))

def major_smooth(select_feature_list,N=50,majoy_rate=0.8):
    thr = np.floor(N*majoy_rate)
    count_dict = {}
    for feature_sets in select_feature_list:
        for select_feature in feature_sets:
            if select_feature in count_dict.keys():
                count_dict[select_feature] += 1
            else:
                count_dict[select_feature] = 1

    smooth_feature = [feature for (feature,counts) in count_dict.items() if counts>=thr]
    print("with major_rate=%.2f, %d / %d features are smoothly seleted!"%(majoy_rate,len(smooth_feature),len(count_dict.items())))
    return smooth_feature
    
def seperate_smooth(select_feature_list_gene,select_feature_list_lip,majoy_rate_gene=0.1,majoy_rate_lip=0.1):
    count_dict_gene = {}
    for feature_sets in select_feature_list_gene:
        for select_feature in feature_sets:
            if select_feature in count_dict_gene.keys():
                count_dict_gene[select_feature] += 1
            else:
                count_dict_gene[select_feature] = 1
    count_dict_lip = {}
    for feature_sets in select_feature_list_lip:
        for select_feature in feature_sets:
            if select_feature in count_dict_lip.keys():
                count_dict_lip[select_feature] += 1
            else:
                count_dict_lip[select_feature] = 1

    thr_gene = np.quantile(list(count_dict_gene.values()),majoy_rate_gene)
    thr_lip = np.quantile(list(count_dict_lip.values()),majoy_rate_lip)

    smooth_feature_gene = [feature for (feature,counts) in count_dict_gene.items() if counts>=thr_gene]
    smooth_feature_lip = [feature for (feature,counts) in count_dict_lip.items() if counts>=thr_lip]

    print("for gene, with major_rate=%.2f, %d / %d features are smoothly seleted!"%(majoy_rate_gene,len(smooth_feature_gene),len(count_dict_gene.items())))
    print("for lip, with major_rate=%.2f, %d / %d features are smoothly seleted!"%(majoy_rate_lip,len(smooth_feature_lip),len(count_dict_lip.items())))

    return smooth_feature_gene + smooth_feature_lip

def my_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def baseline_lr(data_dict):
    clf = Ridge(alpha=1,fit_intercept=False)
    clf.fit(data_dict['X_tr'], data_dict['y_nor_tr'])
    predict = clf.predict(data_dict['X_test']).squeeze()*data_dict['y_std_tr'] + data_dict['y_mean_tr']
    rmse = np.sqrt(np.mean((np.exp(predict)-np.exp(data_dict['y_test'].squeeze()))**2))
    print('with lr, rmse is %.5f'%(rmse))
    return clf.coef_,rmse

def baseline_lrCV(data_dict):
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],fit_intercept=False)
    clf.fit(data_dict['X_tr'], data_dict['y_nor_tr'])
    predict = clf.predict(data_dict['X_test']).squeeze()*data_dict['y_std_tr'] + data_dict['y_mean_tr']
    rmse = np.sqrt(np.mean((np.exp(predict)-np.exp(data_dict['y_test'].squeeze()))**2))
    print('with lrCV, rmse is %.5f'%(rmse))
    return clf.coef_,rmse

def baseline_esnet(data_dict):
    clf = ElasticNet(alpha = 0.2,l1_ratio =0.1,fit_intercept=False)
    clf.fit(data_dict['X_tr'], data_dict['y_nor_tr'])
    predict = clf.predict(data_dict['X_test']).squeeze()*data_dict['y_std_tr'] + data_dict['y_mean_tr']
    rmse = np.sqrt(np.mean((np.exp(predict)-np.exp(data_dict['y_test'].squeeze()))**2))
    print('with elanet, rmse is %.5f'%(rmse))
    return clf.coef_,rmse

def baseline_esnetCV(data_dict):
    clf = ElasticNetCV(l1_ratio =[.001,.01, .1, .5, .7, .9, .95, .99, 1], cv=5,fit_intercept=False)
    clf.fit(data_dict['X_tr'], data_dict['y_nor_tr'])
    predict = clf.predict(data_dict['X_test']).squeeze()*data_dict['y_std_tr'] + data_dict['y_mean_tr']
    rmse = np.sqrt(np.mean((np.exp(predict)-np.exp(data_dict['y_test'].squeeze()))**2))
    print('with elanetCV, rmse is %.5f'%(rmse))
    return clf.coef_,rmse

def baseline_justmean(data_dict):
    predict =  data_dict['y_mean_tr']
    rmse = np.sqrt(np.mean((np.exp(predict)-np.exp(data_dict['y_test'].squeeze()))**2))
    print('with just-mean, rmse is %.5f'%(rmse))
    return rmse

def baseline_LogitLR(data_dict,C=0.5):

    clf = LogisticRegression(random_state=0,penalty='l2',C=C).fit(data_dict['X_tr'],data_dict['y_tr'])
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])
    y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for LogitLR: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'auc':auc,'fpr':fpr,'clf':clf}
    return result_dict

def baseline_LogitLR_leaveone(data_dict,C=0.5):

    clf = LogisticRegression(random_state=0,penalty='l2',C=C).fit(data_dict['X_tr'],data_dict['y_tr'])
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])

    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])


    # print('for LogitLR: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'clf':clf}
    return result_dict

def baseline_LogitElsnet(data_dict,C=0.8,l1_ratio=0.005):

    clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',l1_ratio=l1_ratio,C=C).fit(data_dict['X_tr'],data_dict['y_tr'])
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])
    y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for LogitElsnet: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'auc':auc,'fpr':fpr,'clf':clf}
    return result_dict

def baseline_LogitElsnet_leaveone(data_dict,C=0.8,l1_ratio=0.005):

    clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',l1_ratio=l1_ratio,C=C).fit(data_dict['X_tr'],data_dict['y_tr'])
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])

    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])

    result_dict = {'acr':acr,'clf':clf}
    return result_dict

def baseline_SVM(data_dict):

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True)).fit(data_dict['X_tr'],data_dict['y_tr'])
    
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])
    y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for LogitElsnet: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'auc':None,'fpr':None,'clf':clf}
    return result_dict

def baseline_SVM_leaveone(data_dict):

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(data_dict['X_tr'],data_dict['y_tr'])
    
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])

    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])

    # print('for LogitElsnet: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'clf':clf}
    return result_dict

def baseline_LogitElsnetCV(data_dict,C=0.8,l1_ratio=0.005):
    
    clf = LogisticRegressionCV(l1_ratios =[.001,.01, .1, .5, .7, .9, .95, .99, 1],\
        penalty='elasticnet',solver='saga',fit_intercept=False).fit(data_dict['X_tr'],data_dict['y_tr'])# clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])
    y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for LogitElsnetCV: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'auc':auc,'fpr':fpr,'clf':clf}
    return result_dict

def baseline_RanForest(data_dict):

    clf = RandomForestClassifier(random_state=0).fit(data_dict['X_tr'],data_dict['y_tr'])
    # clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',C=0.5,l1_ratio=l1_ratio).fit(x,y)

    y_pred = clf.predict(data_dict['X_test'])
    y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for RandomForest: accuracy is %f, auc is %f,  fpr is %f'%(acr,auc,fpr))
    result_dict = {'acr':acr,'auc':auc,'fpr':fpr,'clf':clf}
    return result_dict

def baseline_justmode(data_dict):

    y_pred = scipy.stats.mode(data_dict['y_tr'])[0][0] * np.ones(len(data_dict['y_test']))
    # y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    # tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    # fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    # auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    # print('for justmode: accuracy is %f, fpr is %f'%(acr,fpr))
    # result_dict = {'acr':acr,'fpr':fpr,}
    result_dict = {'acr':acr}

    return result_dict

def baseline_random(data_dict):

    y_pred =  np.random.binomial(size=len(data_dict['y_test']), n=1, p= 0.5)
    # y_pred_prob = clf.predict_proba(data_dict['X_test'])

    # print(y_pred)
    # print(y_test)
    # tn, fp, fn, tp = confusion_matrix(data_dict['y_test'], y_pred).ravel()

    # fpr = fp/(fp+tn)
    acr = (y_pred == data_dict['y_test']).sum()/len(data_dict['y_test'])
    # auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    # print('for random-guess: accuracy is %f, fpr is %f'%(acr,fpr))
    # result_dict = {'acr':acr,'fpr':fpr,}
    result_dict = {'acr':acr}

    return result_dict

def baseline_Gibbs_zhe(data_dict,hyper_paras):
    
    
    zhe_model = GibbsSampling3(data_dict['X_tr'],data_dict['y_tr'],hyper_paras['beta'],hyper_paras['r1'], hyper_paras['MAX_NUMBER'],hyper_paras['BURNING'],hyper_paras['r0'])
    w_mean = zhe_model[4][:,0]
    pred_test_porb = norm.cdf(np.dot(data_dict['X_test'],w_mean),loc=0,scale=1)

    pred_test = np.where(pred_test_porb>0.5,1,0)
    acr = (pred_test == data_dict['y_test']).sum()/len(data_dict['y_test'])
    auc = roc_auc_score(data_dict['y_test'], pred_test_porb)



    # auc = roc_auc_score(data_dict['y_test'], y_pred_prob[:,1])

    # acr_array[i] = acr
    # auc_array[i] = auc
    # fpr_array[i] = fpr
    print('for Gibbs-zhe: accuracy is %f, auc is %f'%(acr,auc))
    result_dict = {'acr':acr,'auc':auc,'model':zhe_model}
    return result_dict

def logis_regression_5(X,Y):
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        test_score = clf.score(X_test,y_test)
        # print(test_score)
        loss.append(test_score)
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()



def logis_lasso_regression_5(X,Y,l1_ratio=0.1):
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga',l1_ratio=l1_ratio
).fit(X_train, y_train)
        test_score = clf.score(X_test,y_test)
        # print(test_score)
        loss.append(test_score)
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()



def kernel_svm_regression_5(X,Y,kernel='rbf'):
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel=kernel))

        clf.fit(X_train, y_train)
        test_score = clf.score(X_test,y_test)
        # print(test_score)
        loss.append(test_score)
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()


def gpr_regression_5(X,Y,kernel=DotProduct() + WhiteKernel()):
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    loss = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


        clf = GaussianProcessRegressor(kernel=kernel,random_state=0)

        clf.fit(X_train, y_train)
        # test_score = clf.score(X_test,y_test)
        pred_y = np.where(clf.predict(X_test)>0.5,1,0)

        # print(test_score)
        loss.append(accuracy_score(y_test,pred_y))
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()

def random_forest_regression_5(X,Y):
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    loss = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = RandomForestClassifier(random_state=0)

        clf.fit(X_train, y_train)
        test_score = clf.score(X_test,y_test)
        # print(test_score)
        loss.append(test_score)
    # print('mean of acr: %.3f'%(np.array(loss).mean()))
    return np.array(loss).mean()