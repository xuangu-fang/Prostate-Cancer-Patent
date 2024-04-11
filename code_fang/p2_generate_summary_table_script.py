import numpy as np
import pandas as pd
import pathlib

# P2 results summary

method_index = ['logistic reg','logistic reg-lasso',\
    'linear-kernel svm','RBF-kernel svm','RBF-kernel GPR','matern-kernel GPR' ]
mertric_list = ['accuracy','auc', 'precision','recall','specificity','NPV']
seed_list = [123,321,132,231,123]

# targets of P2

target ='gap_surv_time_class'
sub_target1 = '_early_death'
sub_target2 = '_long_survive'




# target_list = [target+sub_target1 ,target+sub_target2]
target_list = [target+sub_target1]



metric_name = mertric_list[1]
path_str = './result_log_new/P2/' + target_list[0] + '/' + metric_name 
file_name = path_str+'/P2_pred_'+target_list[0]+ '_'+ metric_name+ str(123)+ '.csv'
example_table = pd.read_csv(file_name)
feature_name = example_table.columns[1:]

for target in target_list:

    logis_table_mean = pd.DataFrame(index =feature_name, columns = mertric_list)
    logis_table_std = pd.DataFrame(index =feature_name, columns = mertric_list)

    logis_lasso_table_mean = pd.DataFrame(index =feature_name, columns = mertric_list)
    logis_lasso_table_std = pd.DataFrame(index =feature_name, columns = mertric_list)

    for metric_name in mertric_list:
        path_str = './result_log_new/P2/' + target + '/' + metric_name 
        
        table_list = []

        for rand_seed in seed_list:
            file_name = path_str+'/P2_pred_'+target+ '_'+ metric_name+ str(rand_seed)+ '_binary.csv'
            table_list.append(pd.read_csv(file_name))

        df_concat = pd.concat(table_list)
        by_method = df_concat.groupby('Unnamed: 0')
        df_means = by_method.mean().reindex(method_index)
        df_std = by_method.std().reindex(method_index)
    
        with pd.ExcelWriter(path_str+'/P2_pred_'+target+ '_'+ metric_name+'_summary_binary.xlsx') as writer:
            df_means.to_excel(writer,sheet_name="mean",index=True)
            df_std.to_excel(writer,sheet_name="std",index=True)
        
        logis_table_mean[metric_name] = df_means.loc['logistic reg']
        logis_table_std[metric_name] = df_std.loc['logistic reg']

        logis_lasso_table_mean[metric_name] = df_means.loc['logistic reg-lasso']
        logis_lasso_table_std[metric_name] = df_std.loc['logistic reg-lasso']

    with pd.ExcelWriter('./result_log_new/P2/' + str(target)+'_logit_binary.xlsx') as writer:
        logis_table_mean.to_excel(writer,sheet_name="mean",index=True)
        logis_table_std.to_excel(writer,sheet_name="std",index=True)

    with pd.ExcelWriter('./result_log_new/P2/' + str(target)+'_logit_lasso_binary.xlsx') as writer:
        logis_lasso_table_mean.to_excel(writer,sheet_name="mean",index=True)
        logis_lasso_table_std.to_excel(writer,sheet_name="std",index=True)


    
