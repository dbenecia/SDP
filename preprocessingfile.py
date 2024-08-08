# -*- coding: utf-8 -*-
"""
Created on Jun 4 2024

@author: DX
"""

#Software Defect Prediction 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
import matplotlib.pyplot as plt
from scipy.io import arff

#add dataset normalization and feature selection function

def mapit(vector):

    s = np.unique(vector)

    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    vector=vector.map(mapping)
    return vector

def my_sdp_preprocessor(datafilename_as_csv_inquotes,directory,dataset_name,format):
    if(format == '.csv'):
        original_data = pd.read_csv(datafilename_as_csv_inquotes)
        original_data.isnull().values.any() #Gives false ie:No null value in dataset
        original_data = original_data.fillna(value=False)
        original_X = pd.DataFrame(original_data.drop(['bug'],axis=1))
        original_Y = original_data['bug']
        original_Y = pd.DataFrame(original_Y)
    elif(format == '.arff'):
        original_data, meta = arff.loadarff(directory + dataset_name + '.arff')
        original_X =  pd.DataFrame(original_data)
        original_Y = original_X['Defective']
        original_Y = mapit(original_Y)
        del original_X['Defective']
    
    
    # #changing categorical into numerical values 将n个类别编码为[0,n-1]之间的整数
    from sklearn.preprocessing import LabelEncoder
    encoder_y=LabelEncoder() 
    original_Y=encoder_y.fit_transform(original_Y)
    
    x_train1, x_test, y_train1, y_test= train_test_split(original_X, original_Y, test_size = .1,
                                                              random_state=12) # 将数据拆分为训练集90%和测试集10%
    print(y_train1.shape)

    sm = SMOTE(random_state=12, sampling_strategy = 1.0)
    x, y = sm.fit_resample(x_train1, y_train1)
    y_train2 = pd.DataFrame(y, columns=['bug'])
    x_train2 = pd.DataFrame(x, columns=original_X.columns)
    
    
    x_train, x_val, y_train, y_val= train_test_split(x_train2, y_train2, test_size = .1,
                                                              random_state=12)
    
    combined_training_data = x_train.copy()
    combined_training_data['bug'] = y_train
   
    import seaborn as sns
    corr = combined_training_data.corr()
    
    return original_X, x_train,x_test,x_val,y_train,y_test,y_val 

