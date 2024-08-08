# -*- coding: utf-8 -*-
"""
Created on Jun 4 2024

@author: DX
"""
import pandas as pd
import preprocessingfile as preprocess
import time
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, Embedding, LSTM, Conv1D, MaxPool1D, GRU
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf

def NN(original_X, x_train,x_test,x_val,y_train,y_test,y_val):   
    # Importing the Keras libraries and packages
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    start = time.process_time()
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer 
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = len(original_X.columns)))
    # Adding the first hidden layer
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Adding the second hidden layer
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    
    tf.keras.utils.plot_model(classifier, to_file='./mlp.png', show_shapes=True)
    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

    classifier.fit(x_train, y_train, batch_size=8, epochs=50, verbose=1)

    print(x_train.shape)
    print(y_train.shape)

    #Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = classifier.predict(x_val)
    y_pred = (y_pred > 0.5)    
    y_pred = pd.DataFrame(y_pred, columns=['bug'])

    end = time.process_time()
    duration = end-start
    print( ' Running time of MLP: %s ' %(end-start))
    return classifier, duration

def cnn(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    
    #create model
    start = time.process_time()
    x_train_matrix = x_train.values
    x_test_matrix = x_test.values
    y_train_matrix = y_train.values
    # y_test_matrix = y_test.values
    
    ytrainseries = y_train['bug']
    yvalseries = y_val['bug']

    
    img_rows, img_cols = 1,len(original_X.columns)
    
    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_test1 = x_test_matrix.reshape(x_test_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential() #搭建NN
    #add model layers 3
    model.add(Conv2D(128, kernel_size=1, activation='relu',input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=1, activation='relu'))
   

    model.add(Flatten()) 
    #第一个全连接层
    model.add(Dense(128, activation='relu'))
    #第二个全连接层
    model.add(Dense(1, activation='sigmoid'))
    
    tf.keras.utils.plot_model(model, to_file='./cnn.png', show_shapes=True)
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    #train the model
    model.fit(x_train1, y_train_matrix, batch_size=8, epochs=50, verbose=1)
    y_pred = model.predict(x_test1)>0.5
    y_pred_df = pd.DataFrame(y_pred)
    
    end = time.process_time()
    duration = end-start
    print( ' Running time of CNN: %s ' %(end-start))
    return model, duration   


from sklearn.ensemble import RandomForestClassifier    
def random_forest(original_X,x_train,x_test,x_val,y_train,y_test,y_val):
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    clf.fit(x_train, y_train)
    return clf

from sklearn.svm import SVC
def svm(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    
    start = time.process_time()

    clf =  SVC(kernel='rbf', class_weight='balanced') # linear + balanced: code hangs

    clf.fit(x_train, y_train)
    end = time.process_time()
    duration = end-start
    print( ' Running time of SVM: %s  ' %(end-start))
    return clf, duration


from sklearn.naive_bayes import GaussianNB
# 如果特征是数值的，最好是正态分布的数值的
def gaussiannb(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    start = time.process_time()
    clf = GaussianNB(var_smoothing = 1e-9)
    clf.fit(x_train, y_train)
    end = time.process_time()
    duration = end-start
    print( ' Running time of NB: %s ' %(end-start))
    return clf 


from sklearn.linear_model import LogisticRegression
def lr(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
      
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)
    return clf 

from sklearn.tree import DecisionTreeClassifier 
def decisiontree(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
     
    start = time.process_time()
    # clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=30) # c4.5
    clf = DecisionTreeClassifier(min_samples_leaf=50, criterion = 'gini') #创建 CART 决策树模型,不传任何参数，就是默认使用基尼系数来计算
    clf.fit(x_train, y_train)
    end = time.process_time()
    duration = end-start
    print( ' Running time of DT: %s ' %(end-start))
    return clf, duration   

from sklearn.neighbors import KNeighborsClassifier
def knn(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    start = time.process_time()
    clf=KNeighborsClassifier(n_neighbors=6)
    clf.fit(x_train, y_train)
    end = time.process_time()
    duration = end-start
    print( ' Running time of KNN: %s ' %(end-start))
    return clf, duration    

import xgboost as xgb
def xgbclassifier(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    clf=xgb.XGBClassifier()
    clf.fit(x_train, y_train)
    return clf


from sklearn.cluster import KMeans
def kmeans(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    start = time.process_time()
    clf=KMeans(n_clusters=5, random_state=0)
    clf.fit(x_train, y_train)
    end = time.process_time()
    duration = end-start
    print( ' Running time of KMEANS: %s ' %(end-start))
    return clf


from mlxtend.classifier import StackingClassifier
def stacking(original_X, x_train,x_test,x_val,y_train,y_test,y_val):
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
    sclf.fit(x_train, y_train)
    return sclf

