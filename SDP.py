# -*- coding: utf-8 -*-
"""
Created on Jun  4 15:32:16 2024

@author: DX
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, roc_curve
# from imblearn.over_sampling import SMOTE
from sklearn import model_selection
import preprocessingfile as preprocess
import mymodels
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import specificity_score
from tensorflow.keras.utils import plot_model

start = time.time()
dataset_name = 'jm1' # xerces-1.2
format = '.csv'
directory = './datasets/'
data = directory + dataset_name + format
original_X, x_train,x_test,x_val,y_train,y_test,y_val = preprocess.my_sdp_preprocessor(data,directory,dataset_name,format)
all_data = [original_X, x_train, x_test, x_val, y_train,y_test,y_val]

st_clf = mymodels.stacking(*all_data)
lr_clf = mymodels.lr(*all_data)
nn_clf, nn_duration = mymodels.NN(*all_data)
cnn_clf, cnn_duration = mymodels.cnn(*all_data) 
knn_clf, dt_duration = mymodels.knn(*all_data)
svm_clf, svm_duration = mymodels.svm(*all_data)
dt_clf, dt_duration = mymodels.decisiontree(*all_data)
nb_clf = mymodels.gaussiannb(*all_data)
xgb_clf = mymodels.xgbclassifier(*all_data)
rf_clf = mymodels.random_forest(*all_data)

def g_mean_metric(y_true, y_pred):
    y_pred = np.array([1 if x >= 0.5 else 0 for x in y_pred])

    recall = recall_score(y_true, y_pred,average='macro')
    
    i = np.where(y_pred == 0)[0]
    i2 = np.where(y_true == 0)[0]
    tn = float(np.intersect1d(i, i2).size) 

    i = np.where(y_pred == 1)[0]
    i2 = np.where(y_true == 0)[0]
    fp = float(np.intersect1d(i, i2).size)

    specificity = (tn / (tn + fp))

    mult = recall * specificity
    gmean = np.sqrt(mult)
    
    return gmean, specificity

def getBalance(y_true,y_pred):
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    print('fpr=', fpr)
    FPR = fpr[int(len(fpr)/2)]
    TPR = tpr[int(len(fpr)/2)]
    print('FPR=', FPR)
    Balance = 1 - np.sqrt(((1 - TPR) ** 2 + FPR ** 2)/2)
    return Balance

def return_metrics(model): 
    if (model == nn_clf):
        y_pred_on_val = model.predict(x_val)>0.5
        y_pred_on_test = model.predict(x_test)>0.5
    elif (model == cnn_clf):
        x_val_matrix = x_val.values
        x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], 1, len(x_val.columns), 1)
        y_pred_on_val = model.predict(x_val1)>0.5
        x_test_matrix = x_test.values
        x_test1 = x_test_matrix.reshape(x_test_matrix.shape[0], 1, len(x_test.columns), 1)
        y_pred_on_test = model.predict(x_test1)>0.5

    else:
        y_pred_on_val = model.predict(x_val)
        y_pred_on_test = model.predict(x_test)
   
    print('-------------------------------------------------')
    print()    
    # print('******', str(model), '******')   
    print('||Validation Set||')
    cm=confusion_matrix(y_val,np.round(y_pred_on_val))
    print(cm)
    print('Accuracy:',accuracy_score(y_val,np.round(y_pred_on_val)))
    print('f1_score:', f1_score(y_val,np.round(y_pred_on_val),average='weighted'))
    print('Precision:', precision_score(y_val,np.round(y_pred_on_val),average='weighted'))
    print('Recall:', recall_score(y_val, np.round(y_pred_on_val),average='weighted'))
    print('ROC_AUC:',roc_auc_score(y_val,np.round(y_pred_on_val),average='weighted'))
    gmean = geometric_mean_score(y_val, np.round(y_pred_on_val), average='weighted')
    specificity = specificity_score(y_val, np.round(y_pred_on_val), average='weighted')
    mcc = matthews_corrcoef(y_val,np.round(y_pred_on_val))
    balance = getBalance(y_val,np.round(y_pred_on_val))
    print("gmean=", gmean)
    print("specificity=", specificity)
    print('mcc=', mcc)
    print('balance=', balance)
    print('||Test Set||')
    cm=confusion_matrix(y_test,np.round(y_pred_on_test))
    print(cm)
    print('Accuracy:',accuracy_score(y_test,np.round(y_pred_on_test)))
    print('f1_score:', f1_score(y_test,np.round(y_pred_on_test),average='weighted'))
    print('Precision:', precision_score(y_test,np.round(y_pred_on_test),average='weighted'))
    print('Recall:', recall_score(y_test, np.round(y_pred_on_test),average='weighted'))
    print('ROC_AUC:',roc_auc_score(y_test,np.round(y_pred_on_test),average='weighted'))
    gmean = geometric_mean_score(y_test,np.round(y_pred_on_test), average='weighted')
    specificity = specificity_score(y_test,np.round(y_pred_on_test), average='weighted')
    mcc = matthews_corrcoef(y_test,np.round(y_pred_on_test))
    balance = getBalance(y_test,np.round(y_pred_on_test))
    print('balance=', balance)
    print("gmean=", gmean)
    print("specificity=", specificity)
    print('mcc=', mcc)
    print('-------------------------------------------------')
    print()
    
    accuracy = accuracy_score(y_test,np.round(y_pred_on_test))
    precision = precision_score(y_test,np.round(y_pred_on_test),average='weighted')
    recall = recall_score(y_test,np.round(y_pred_on_test),average='weighted')
    fscore = f1_score(y_test,np.round(y_pred_on_test),average='weighted')
    auc = roc_auc_score(y_test,np.round(y_pred_on_test),average='weighted')
    gmean = geometric_mean_score(y_test,np.round(y_pred_on_test), average='weighted')
    specificity = specificity_score(y_test,np.round(y_pred_on_test), average='weighted')
    mcc = matthews_corrcoef(y_test,np.round(y_pred_on_test))
    balance = getBalance(y_test,np.round(y_pred_on_test))
    
    return accuracy,precision,recall,fscore,auc, gmean,specificity,cm,mcc,balance
    
accuracy_nb,precision_nb,recall_nb,fscore_nb,auc_nb,gmean_nb,specificity_nb,cm_nb,mcc_nb,balance_nb=return_metrics(nb_clf)
accuracy_nn,precision_nn,recall_nn,fscore_nn,auc_nn,gmean_nn,specificity_nn,cm_nn,mcc_nn,balance_nn=return_metrics(nn_clf)
accuracy_cnn,precision_cnn,recall_cnn,fscore_cnn,auc_cnn,gmean_cnn,specificity_cnn,cm_cnn,mcc_cnn,balance_cnn=return_metrics(cnn_clf)
accuracy_svm,precision_svm,recall_svm,fscore_svm,auc_svm,gmean_svm,specificity_svm,cm_svm,mcc_svm,balance_svm=return_metrics(svm_clf)
accuracy_dt,precision_dt,recall_dt,fscore_dt,auc_dt,gmean_dt,specificity_dt,cm_dt,mcc_dt,balance_dt=return_metrics(dt_clf)
accuracy_knn,precision_knn,recall_knn,fscore_knn,auc_knn,gmean_knn,specificity_knn,cm_knn,mcc_knn,balance_knn=return_metrics(knn_clf)
accuracy_xgb,precision_xgb,recall_xgb,fscore_xgb,auc_xgb,gmean_xgb,specificity_xgb,cm_xgb,mcc_xgb,balance_xgb=return_metrics(xgb_clf)
accuracy_lr,precision_lr,recall_lr,fscore_lr,auc_lr,gmean_lr,specificity_lr,cm_lr,mcc_lr,balance_lr=return_metrics(lr_clf)
accuracy_rf,precision_rf,recall_rf,fscore_rf,auc_rf,gmean_rf,specificity_rf,cm_rf,mcc_rf,balance_rf=return_metrics(rf_clf)
accuracy_st,precision_st,recall_st,fscore_st,auc_st,gmean_st,specificity_st,cm_st,mcc_st,balance_st=return_metrics(st_clf)

stacking=np.array([mcc_st])#,precision_stacking,recall_stacking,fscore_stacking,auc_stacking,gmean_stacking])
nb=np.array([mcc_nb])#,precision_nb,recall_nb,fscore_nb,auc_nb,gmean_nb])
knn=np.array([mcc_knn])#,precision_knn,recall_knn,fscore_knn,auc_knn,gmean_knn])
lr=np.array([mcc_lr])#,precision_kmeans,recall_kmeans,fscore_kmeans,auc_kmeans,gmean_kmeans])
cnn=np.array([mcc_cnn])#,precision_cnn,recall_cnn,fscore_cnn,auc_cnn,gmean_cnn])
nn=np.array([mcc_nn])#,precision_nn,recall_nn,fscore_nn,auc_nn,gmean_nn])
svm=np.array([mcc_svm])#,precision_svm,recall_svm,fscore_svm,auc_svm,gmean_svm,])
dt=np.array([mcc_dt])#,precision_dt,recall_dt,fscore_dt,auc_dt,gmean_dt])
xgb=np.array([mcc_xgb])
rf=np.array([mcc_rf])

x=np.arange(len(nb))

import matplotlib.patches as mpatches
bar_width=0.6
bar_width1=1

bar1 = plt.bar(x,lr,width=bar_width,color='green',zorder=2)
bar2 = plt.bar(x+bar_width1,nb,width=bar_width,color='orange',zorder=2)
bar3 = plt.bar(x+bar_width1*2,knn,width=bar_width,color='red',zorder=2)
bar4 = plt.bar(x+bar_width1*3,dt,width=bar_width,color='darkviolet',zorder=2)
bar5 = plt.bar(x+bar_width1*4,svm,width=bar_width,color='cyan',zorder=2)
bar6 = plt.bar(x+bar_width1*5,nn,width=bar_width,color='magenta',zorder=2)
bar7 = plt.bar(x+bar_width1*6,cnn,width=bar_width,color='blue',zorder=2)
bar8 = plt.bar(x+bar_width1*7,rf,width=bar_width,color='darkorange',zorder=2)
bar9 = plt.bar(x+bar_width1*8,xgb,width=bar_width,color='lawngreen',zorder=2)
bar10 = plt.bar(x+bar_width1*9,stacking,width=bar_width,color='gray',zorder=2)

Y = [lr, nb, knn, dt, svm, nn, cnn, rf, xgb, stacking]
print("Y=", Y)
    
x=np.arange(10)
plt.xticks(x,["LR","NB","KNN","DT","SVM","MLP","CNN","RF","XGB","Stacking"],size=20)
plt.title(dataset_name+': Comparison of MCC among 10 Methods',fontsize=20)

plt.grid(axis='y')
from matplotlib.ticker import MultipleLocator
ymajorLocator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.grid(True, which='major') #y坐标轴的网格使用主刻度
plt.tick_params(labelsize=20)

end = time.time()
print( ' Total running time : %s ' %(end-start))