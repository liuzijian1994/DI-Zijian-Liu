
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import model_selection, ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# In[2]:


dir_path=r'./data'

def import_data(path):
    imdict = {}
    for file in os.listdir(path):
        imdict[file[:-4]] = pd.read_csv(path+r'\\'+file,low_memory = False)
    return imdict



data = pd.read_csv(dir_path+'\\LoanStats_securev1_2018Q1.csv',low_memory = False,skiprows = 1)



cate_list = list(data.select_dtypes(include='object').columns)
conti_list = list(data.select_dtypes(include='float64').columns)



def table_cate(df,col):
    t = pd.pivot_table(df[[col, 'loan_status']], index=col, columns=['loan_status'], aggfunc=len)
    t['ratio'] = t['Charged Off'].values / (t['Current'].values + t['Charged Off'].values+t['Fully Paid'].values)
    t = t.sort_values(by = 'ratio',ascending=False)
    return t



def plot_cate(df,col_list):
    for name in col_list:
        p = pd.pivot_table(df[[name, 'loan_status']], index=df[name], columns=['loan_status'], aggfunc=len)
        p.plot(kind='bar')



def plot_conti(df,col):
    plt.hist(df[df.loan_status=='Fully Paid'].loc[:, col], color='green')
    plt.hist(df[df.loan_status=='Charged Off'].loc[:, col],color='red')
    plt.title(col)
    plt.legend(['good', 'bad'])


# NULL



def null_check(df,coltype,top):
    target_cols = df.select_dtypes(include=coltype).columns
    return(((df[target_cols].isnull().sum()/df.shape[0]).sort_values(ascending=False))[:top],)



data_clean = data.drop(columns=['member_id', 'desc'])



hardship_related_cols = ['hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd'                         ,'hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount'                         ,'hardship_last_payment_amount','debt_settlement_flag','debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term']
data_clean = data_clean.drop(columns=hardship_related_cols)



data_clean = data_clean.loc[~data_clean['application_type'].isnull()]
data_ind_c = data_clean.loc[data_clean['application_type'] =='Individual']



joint_related_cols = ['sec_app_earliest_cr_line','verification_status_joint','sec_app_mths_since_last_major_derog','sec_app_inq_last_6mths','sec_app_collections_12_mths_ex_med','annual_inc_joint'                      ,'dti_joint','revol_bal_joint','sec_app_fico_range_high','sec_app_fico_range_low','sec_app_mort_acc'                      ,'sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il'                      ,'sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','mths_since_last_record']
data_ind_c = data_ind_c.drop(columns=joint_related_cols)



data_ind_c = data_ind_c.drop(columns=['next_pymnt_d'])
data_ind_c['emp_title'].loc[data_ind_c['emp_title'].isnull()] = 'no data'
data_ind_c['emp_length'].loc[data_ind_c['emp_length'].isnull()] = 'no data'
last_event_cols = ['mths_since_last_major_derog','mths_since_recent_bc_dlq','mths_since_recent_revol_delinq','mths_since_last_delinq','mths_since_recent_inq','mths_since_rcnt_il','mths_since_recent_bc']
for var in last_event_cols:
    data_ind_c[var].loc[data_ind_c[var].isnull()] = -1



data_ind_c['il_util'].loc[(data_ind_c['num_il_tl']!=0) & (data_ind_c['il_util'].isnull())] =data_ind_c['il_util'].loc[(data_ind_c['num_il_tl']!=0)].mean()
data_ind_c['il_util'].loc[data_ind_c['num_il_tl']==0] = 0
data_ind_c['bc_util'].loc[(data_ind_c['num_bc_tl']!=0) & (data_ind_c['bc_util'].isnull())] = data_ind_c['bc_util'].loc[(data_ind_c['num_bc_tl']!=0)].mean()
data_ind_c['bc_util'].loc[data_ind_c['num_bc_tl']==0] = 0
data_ind_c['percent_bc_gt_75'].loc[(data_ind_c['num_bc_tl']!=0) & (data_ind_c['percent_bc_gt_75'].isnull())] = data_ind_c['percent_bc_gt_75'].loc[(data_ind_c['num_bc_tl']!=0)].mean()
data_ind_c['percent_bc_gt_75'].loc[data_ind_c['num_bc_tl']==0] = 0
data_ind_c['bc_open_to_buy'].loc[(data_ind_c['bc_open_to_buy'].isnull())] = data_ind_c['bc_open_to_buy'].mean()
data_ind_c['mo_sin_old_il_acct'].loc[data_ind_c['num_il_tl']==0] = -1



data_ind_c = data_ind_c.drop(columns=['num_tl_120dpd_2m'])
data_ind_c.dropna(axis=0, how='any', inplace=True)


# SPLIT



obj_cols = []
float_cols = []

for col in data_ind_c.columns:
    if (data_ind_c[col].dtypes == 'O') and (col != 'loan_status'):
        obj_cols.append(col)
    if data_ind_c[col].dtypes == 'float64':
        float_cols.append(col)

data_ind_float = data_ind_c[float_cols]
data_ind_object = data_ind_c[obj_cols]
target_df = data_ind_c[['loan_status']]



int_rate_df = data_ind_object[['int_rate']]
revol_util_df = data_ind_object[['revol_util']]
def str_to_float(df,col_name):
    df[col_name] = df[col_name].map(lambda x:x.strip())
    df[col_name] = df[col_name].map(lambda x:float(x[:-1]))
    return df
int_rate_df = str_to_float(int_rate_df,'int_rate')
revol_util_df = str_to_float(revol_util_df,'revol_util')



data_ind_object=data_ind_object.drop(['int_rate','revol_util'],axis=1)
data_ind_float = pd.merge(data_ind_float, int_rate_df, left_index =True,right_index =True)
data_ind_float = pd.merge(data_ind_float, revol_util_df, left_index =True,right_index =True)



data_ind_object=data_ind_object.drop(['id','sub_grade','url','zip_code','addr_state','emp_title','earliest_cr_line','application_type'],axis=1)


# Transformer

STD_scaler = StandardScaler()
STD_scaler.fit(data_ind_float)
nparray_std_float = STD_scaler.transform(data_ind_float)
data_std_float = pd.DataFrame(nparray_std_float, index=data_ind_float.index, columns=data_ind_float.columns)



model_save_path = "./models_save/"
STD_save_path_name=model_save_path+"STD_scaler.m"
joblib.dump(STD_scaler, STD_save_path_name)



std_scaler = joblib.load("./models_save/STD_scaler.m")

test_input_num = std_scaler.transform(data_ind_float)



test_input_num


# Target_transform



target_df['loan_status'].loc[(target_df['loan_status'] == 'Current')|(target_df['loan_status'] == 'Fully Paid')] = 0
target_df['loan_status'].loc[target_df['loan_status'] != 0] = 1


# Cata_transform



trans_cata_df=pd.merge(target_df, data_ind_object, left_index =True,right_index =True)



data_ind_object['emp_length'].loc[(data_ind_object['emp_length']=='< 1 year')|                                  (data_ind_object['emp_length']=='1 year')|(data_ind_object['emp_length']=='2 years')] = 'short time'
data_ind_object['emp_length'].loc[(data_ind_object['emp_length']=='3 years')|                                  (data_ind_object['emp_length']=='4 years')|                                  (data_ind_object['emp_length']=='5 years')|(data_ind_object['emp_length']=='6 years')] = 'medium time'
data_ind_object['emp_length'].loc[(data_ind_object['emp_length']=='7 years')|                                  (data_ind_object['emp_length']=='8 years')|                                  (data_ind_object['emp_length']=='9 years')|(data_ind_object['emp_length']=='10+ years')] = 'long time'



data_ind_object['purpose'].loc[(data_ind_object['purpose']=='credit_card')|                                  (data_ind_object['purpose']=='debt_consolidation')|(data_ind_object['purpose']=='small_business')] = 'credit'
data_ind_object['purpose'].loc[(data_ind_object['purpose']=='car')|                                  (data_ind_object['purpose']=='home_improvement')|(data_ind_object['purpose']=='major_purchase')|                              (data_ind_object['purpose']=='moving')|(data_ind_object['purpose']=='renewable_energy')|                              (data_ind_object['purpose']=='vacation')] = 'spending'
data_ind_object['purpose'].loc[(data_ind_object['purpose']=='house')|                                  (data_ind_object['purpose']=='wedding')] = 'big spending'



data_ind_object=data_ind_object.drop(['issue_d','last_pymnt_d','last_credit_pull_d','title'],axis=1)
trans_cata_df=pd.merge(target_df, data_ind_object, left_index =True,right_index =True)


# Dummy



data_ind_object = pd.get_dummies(data_ind_object,drop_first = True)


# Outlier



obj_cols = []
float_cols = []

for col in data_ind_c.columns:
    if (data_ind_c[col].dtypes == 'O') and (col != 'loan_status'):
        obj_cols.append(col)
    if data_ind_c[col].dtypes == 'float64':
        float_cols.append(col)



clf = IsolationForest(n_estimators =100, behaviour = 'new', max_samples=256, random_state = 42, contamination= 'auto')
preds = clf.fit_predict(data_std_float)
outlier_list = list(np.where(preds == -1))
data_std_float['outlier'] = preds
out_df = data_std_float.loc[data_std_float['outlier']==-1]
nonout_df =data_std_float.loc[data_std_float['outlier']==1]
full_df=pd.merge(target_df, data_std_float, left_index =True,right_index =True)
full_df = pd.merge(full_df, data_ind_object, left_index =True,right_index =True)


# Before Modeling



y_full = target_df
X_full = pd.merge(data_ind_object, data_std_float, left_index =True,right_index =True)



y_numout = full_df[['loan_status']].loc[full_df['outlier']==1]
X_numout = full_df.loc[full_df['outlier']==1].drop(['loan_status'],axis=1)
#89303 records



X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
X_numout_train, X_numout_test, y_numout_train, y_numout_test = train_test_split(X_numout, y_numout, test_size=0.2, random_state=42)



def statistic_model(y_test,y_pred,model,X_test):
    print('result of %s' %model)
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    print("MCC: %f " %matthews_corrcoef(y_test, y_pred))
    print("ACC: %f " %accuracy_score(y_test, y_pred))
    print("precision: %f " %precision_score(y_test, y_pred))
    print("recall: %f " %recall_score(y_test, y_pred))
    y_prob =model.predict_proba(X_test)[:,1]
    print( "AUC:  %f "  %roc_auc_score(y_test,y_prob))


# Naive model



navie_data=np.zeros(len(y_full_test))
naive_pred=pd.DataFrame(navie_data,columns=['loan_status'])




cm_navie = confusion_matrix(y_full_test, naive_pred)

print(cm_navie)

print("MCC: %f " %matthews_corrcoef(y_full_test, naive_pred))
print("ACC: %f " %accuracy_score(y_full_test, naive_pred))
print("precision: %f " %precision_score(y_full_test, naive_pred))
print("recall: %f " %recall_score(y_full_test, naive_pred))
print( "AUC:  %f "  %roc_auc_score(y_full_test,naive_pred))


# Logistic Regression



LR_full_b=LogisticRegression(class_weight='balanced')
lr_f_b_m = LR_full_b.fit(X_full_train, y_full_train)
y_LR_pred = lr_f_b_m.predict(X_full_test)
#full dataset balanced weight



statistic_model(y_full_test,y_LR_pred,lr_f_b_m,X_full_test)
'''
Best Penalty: l2
Best C: 1.0
-1265.0696976184845
'''


# GB trees



OS = RandomOverSampler(ratio='auto', random_state=0)
osx_full, osy_full = OS.fit_sample(X_full_train, y_full_train)



GB =  GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.5, loss='deviance', max_depth=3,
                           max_features='auto', max_leaf_nodes=22,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=5,
                           min_weight_fraction_leaf=0.0, n_estimators=120,
                           n_iter_no_change=None, presort='auto',
                           random_state=42, subsample=0.99, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
GB.fit(osx_full, osy_full)
y_GB_pred = GB.predict(X_full_test)
#OverSampling


statistic_model(y_full_test,y_GB_pred,GB,X_full_test)


# NN



NN = MLPClassifier(solver='sgd',momentum=0.95, learning_rate_init=0.01,hidden_layer_sizes=64,batch_size=512,alpha=0.0001,random_state=42)
NN.fit(X_full_train, y_full_train)
y_NN_pred = NN.predict(X_full_test)



statistic_model(y_full_test,y_NN_pred,NN,X_full_test)


# RF


RF = RandomForestClassifier(oob_score = False,random_state=42,n_estimators=460,min_samples_split=2,min_samples_leaf=1,max_features='auto',max_depth=90)
RF.fit(X_full_train, y_full_train)
y_RF_pred = RF.predict(X_full_test)



statistic_model(y_full_test,y_RF_pred,RF,X_full_test)


# Ensemble Model



def statistic_E_model(y_test,y_prob_pred):
    print('prediction model')
    y_pred = y_prob_pred.copy()
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    print("MCC: %f " %matthews_corrcoef(y_test, y_pred))
    print("ACC: %f " %accuracy_score(y_test, y_pred))
    print("precision: %f " %precision_score(y_test, y_pred))
    print("recall: %f " %recall_score(y_test, y_pred))
    print( "AUC:  %f "  %roc_auc_score(y_test,y_prob_pred))


# In[48]:


y_LR_prob = lr_f_b_m.predict_proba(X_full_test)
y_GB_prob = GB.predict_proba(X_full_test)
y_NN_prob = NN.predict_proba(X_full_test)
y_RF_prob = RF.predict_proba(X_full_test)


# In[49]:



EMP3 = (y_LR_prob[:,1]*0.4+y_RF_prob[:,1]*0.1+y_NN_prob[:,1]*0.4+y_GB_prob[:,1]*0.1)




# In[55]:


model_save_path = "./models_save/"
LR_save_path_name=model_save_path+"LR_"+"train_model.m"
joblib.dump(lr_f_b_m, LR_save_path_name)
GB_save_path_name=model_save_path+"GB_"+"train_model.m"
joblib.dump(GB, GB_save_path_name)
NN_save_path_name=model_save_path+"NN_"+"train_model.m"
joblib.dump(NN, NN_save_path_name)
RF_save_path_name=model_save_path+"RF_"+"train_model.m"
joblib.dump(RF, RF_save_path_name)


# In[57]:


# droped columns, for input purpose.

# In[81]:

'''

useless_cols = ['member_id', 'desc','hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount'                         ,'hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd'                         ,'hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount'                         ,'hardship_last_payment_amount','debt_settlement_flag','debt_settlement_flag_date'                         ,'settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term'               ,'sec_app_earliest_cr_line','verification_status_joint','sec_app_mths_since_last_major_derog'                      ,'sec_app_inq_last_6mths','sec_app_collections_12_mths_ex_med','annual_inc_joint'                      ,'dti_joint','revol_bal_joint','sec_app_fico_range_high','sec_app_fico_range_low','sec_app_mort_acc'                      ,'sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il'                      ,'sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','mths_since_last_record'               ,'next_pymnt_d','num_tl_120dpd_2m','id','sub_grade','url','zip_code'                ,'addr_state','emp_title','earliest_cr_line','application_type','issue_d','last_pymnt_d','last_credit_pull_d','title']


# In[82]:


input_data = data.drop(columns=useless_cols)


# In[107]:


input_data = input_data[:10]


# In[110]:


def str_to_float(df,col_name):
    df[col_name] = df[col_name].map(lambda x:x.strip())
    df[col_name] = df[col_name].map(lambda x:float(x[:-1]))
    return df


# In[133]:


def input_transformation(input_df):
    obj_cols = []
    float_cols = []
    for col in input_df.columns:
        if (input_df[col].dtypes == 'O') and (col != 'loan_status'):
            obj_cols.append(col)
        if input_df[col].dtypes == 'float64':
            float_cols.append(col)
    input_num = input_df[float_cols]
    input_obj = input_df[obj_cols]
    #split


    int_rate = input_obj[['int_rate']]
    revol_util = input_obj[['revol_util']]
    int_rate = str_to_float(int_rate,'int_rate')
    revol_util = str_to_float(revol_util,'revol_util')
    input_obj= input_obj.drop(['int_rate','revol_util'],axis=1)
    input_num = pd.merge(input_num, int_rate, left_index =True,right_index =True)
    input_num = pd.merge(input_num, revol_util, left_index =True,right_index =True)

    std_scaler = joblib.load("./models_save/STD_scaler.m")

    nparray_num = std_scaler.transform(input_num)
    input_num = pd.DataFrame(nparray_num, index=input_num.index, columns=input_num.columns)
    #numerical transformation

    input_obj['emp_length'].loc[(input_obj['emp_length']=='< 1 year')|                                  (input_obj['emp_length']=='1 year')|(input_obj['emp_length']=='2 years')] = 'short time'
    input_obj['emp_length'].loc[(input_obj['emp_length']=='3 years')|                                  (input_obj['emp_length']=='4 years')|                                  (input_obj['emp_length']=='5 years')|(input_obj['emp_length']=='6 years')] = 'medium time'
    input_obj['emp_length'].loc[(input_obj['emp_length']=='7 years')|                                  (input_obj['emp_length']=='8 years')|                                  (input_obj['emp_length']=='9 years')|(input_obj['emp_length']=='10+ years')] = 'long time'

    input_obj['purpose'].loc[(input_obj['purpose']=='credit_card')|                                  (input_obj['purpose']=='debt_consolidation')|(input_obj['purpose']=='small_business')] = 'credit'
    input_obj['purpose'].loc[(input_obj['purpose']=='car')|                                  (input_obj['purpose']=='home_improvement')|(input_obj['purpose']=='major_purchase')|                              (input_obj['purpose']=='moving')|(input_obj['purpose']=='renewable_energy')|                              (input_obj['purpose']=='vacation')] = 'spending'
    input_obj['purpose'].loc[(input_obj['purpose']=='house')|                                  (input_obj['purpose']=='wedding')] = 'big spending'

    input_obj = pd.get_dummies(input_obj,drop_first = True)
    #one hot encoding
    return pd.merge(input_obj, input_num, left_index =True,right_index =True)


# In[134]:


input_transformation(input_data)


# In[135]:


t_data={
       'date':pd.date_range('20000101',periods=10),
       'gender':np.random.randint(0,2,size=10),
       'height':np.random.randint(40,50,size=10),
       'weight':np.random.randint(150,180,size=10)
   }


# In[121]:


test_a = pd.DataFrame(t_data)


# In[122]:


test_a


# In[127]:


t_record = [['2000-01-01',1,45,175]]


# In[128]:


input_num = pd.DataFrame(t_record,columns=test_a.columns)


# In[129]:


input_num


# In[130]:


input_cols = input_data.columns


# In[138]:


def input_data(loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate,
       installment, grade, emp_length, home_ownership, annual_inc,
       verification_status, loan_status, pymnt_plan, purpose, dti,
       delinq_2yrs, fico_range_low, fico_range_high, inq_last_6mths,
       mths_since_last_delinq, open_acc, pub_rec, revol_bal,
       revol_util, total_acc, initial_list_status, out_prncp,
       out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp,
       total_rec_int, total_rec_late_fee, recoveries,
       collection_recovery_fee, last_pymnt_amnt, last_fico_range_high,
       last_fico_range_low, collections_12_mths_ex_med,
       mths_since_last_major_derog, policy_code, acc_now_delinq,
       tot_coll_amt, tot_cur_bal,open_acc_6m, open_act_il,
       open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il,
       il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util,
       total_rev_hi_lim, inq_fi, total_cu_tl, inq_last_12m,
       acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util,
       chargeoff_within_12_mths, delinq_amnt, mo_sin_old_il_acct,
       mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl,
        mort_acc, mths_since_recent_bc, mths_since_recent_bc_dlq,
       mths_since_recent_inq, mths_since_recent_revol_delinq,
       num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl,
       num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl,
       num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_30dpd,
       num_tl_90g_dpd_24m, num_tl_op_past_12m, pct_tl_nvr_dlq,
       percent_bc_gt_75, pub_rec_bankruptcies, tax_liens,
       tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit,
       total_il_high_credit_limit):




    return[loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate,
       installment, grade, emp_length, home_ownership, annual_inc,
       verification_status, loan_status, pymnt_plan, purpose, dti,
       delinq_2yrs, fico_range_low, fico_range_high, inq_last_6mths,
       mths_since_last_delinq, open_acc, pub_rec, revol_bal,
       revol_util, total_acc, initial_list_status, out_prncp,
       out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp,
       total_rec_int, total_rec_late_fee, recoveries,
       collection_recovery_fee, last_pymnt_amnt, last_fico_range_high,
       last_fico_range_low, collections_12_mths_ex_med,
       mths_since_last_major_derog, policy_code, acc_now_delinq,
       tot_coll_amt, tot_cur_bal,open_acc_6m, open_act_il,
       open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il,
       il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util,
       total_rev_hi_lim, inq_fi, total_cu_tl, inq_last_12m,
       acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util,
       chargeoff_within_12_mths, delinq_amnt, mo_sin_old_il_acct,
       mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl,
        mort_acc, mths_since_recent_bc, mths_since_recent_bc_dlq,
       mths_since_recent_inq, mths_since_recent_revol_delinq,
       num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl,
       num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl,
       num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_30dpd,
       num_tl_90g_dpd_24m, num_tl_op_past_12m, pct_tl_nvr_dlq,
       percent_bc_gt_75, pub_rec_bankruptcies, tax_liens,
       tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit,
       total_il_high_credit_limit]
'''
