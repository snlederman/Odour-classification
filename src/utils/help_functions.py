
"""
Created on Sun Dec 19 18:44:02 2021

@author: ariel
help function for the main code
"""
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut


# ------------- split function ----------------------------------------------------------------------------------
def split_once(df,flag,*args):
    if flag:
        X = df.drop(['label','ID','channel','date'],axis = 'columns')
        y = df['label']
        train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df, groups=df['ID']))
        train = df.iloc[train_inds]
        test = df.iloc[test_inds]
        X_train = train.drop(['label','ID','date','channel'],axis = 'columns')
        y_train = train['label']
        X_test = test.drop(['label','ID','date','channel'],axis = 'columns')
        y_test = test['label']
    
    else:
        X = df.drop(['label','ID','channel','date'],axis = 'columns')
        y = df['label']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
    
    return X,y,X_train,X_test,y_train,y_test

#-------------check samples after normalization ------------------------------------------
def plot_check_normalization(X_normalized):
    time_data = pd.read_csv('normalized/time.csv')
    time_data = time_data.iloc[0].values
    plt.figure(figsize=(25, 15))
    color=iter(cm.rainbow(np.linspace(0,1,30)))
    for i in range(30,60):
        c = next(color)
        plt.plot(time_data, X_normalized.values[i,:],c=c,label = i)
        plt.legend()
        plt.title('Data samples')
        plt.xlabel('time [sec]')
        plt.ylabel('Amplitude [mV]')
        plt.show()
    
# -------- confusion matrix of one split data ---------------------------------------------     
def confusion_mat_graph_onesplit(RandomForest,X_train,y_train,X_test,y_test,data_flag,df):
    RandomForest.fit(X_train,y_train)
    y_predicted = RandomForest.predict(X_test)
    plt.figure(figsize = (7,7))
    # cf_matrix = confusion_matrix(y_test,y_predicted)
    # ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    #             fmt='.2%', cmap='Blues')
    cf_matrix = confusion_matrix(y_test,y_predicted,normalize = 'true',labels=['1-Benz','2-Hex','3-Ethyl','4-Rose','5-Lem','6-Ger','7-Cit','8-Van'])
    # cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cf_matrix, annot=None, vmin=0, vmax=1,
            fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix with labels');
    ax.set_xlabel('Predicted odors')
    ax.set_ylabel('Actual odors');
    
    if data_flag == '5 odors':
        ax.xaxis.set_ticklabels(['Benz', 'Cit','Ethyl', 'Geran','Hex'])
        ax.yaxis.set_ticklabels(['Benz', 'Cit','Ethyl', 'Geran', 'Hex'])
    if data_flag == 'mix1':
        ax.xaxis.set_ticklabels(['Benz', 'Cit','Mix1'])
        ax.yaxis.set_ticklabels(['Benz', 'Cit','Mix1'])
    if data_flag == 'mix2':
        ax.xaxis.set_ticklabels(['Ethyl', 'Hex','Mix2'])
        ax.yaxis.set_ticklabels(['Ethyl', 'Hex','Mix2'])
    if data_flag == 'benz hex ethyl':
        ax.xaxis.set_ticklabels(['Benz', 'Ethyl','Hex'])
        ax.yaxis.set_ticklabels(['Benz', 'Ethyl','Hex'])        
    if data_flag == 'cit ger':
        ax.xaxis.set_ticklabels(['Cit', 'Geran'])
        ax.yaxis.set_ticklabels(['Cit', 'Geran'])
    if data_flag == 'test_train 10^0' or  data_flag == 'test_train 10^-1' or data_flag == 'test_train 10^-2' or data_flag == 'test_train 10^-3' or data_flag == 'test_train 10^-4' or data_flag == 'test_train 10^1':
        ax.xaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
        ax.yaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
    Accuracy = RandomForest.score(X_test,y_test)
    print('The Accuracy is: ' + str(round(Accuracy*100)) + '% for ' + str(len(df.index)) + ' samples total')
    
# --------- confusion matrix of the cross validation ---------------------------------------
# def confusion_mat_graph_cv(y,ypred_mean,ypred_mean_index,accuracy_all,df,data_flag):
def confusion_mat_graph_cv(y,ypred,accuracy_all,df,data_flag):
    plt.figure(figsize = (7,7))
    cf_matrix = confusion_matrix(y,ypred)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cmn, annot=None,vmax=1,vmin=0,
            fmt='.2%', cmap='Blues')
    #ax.set_title('Confusion Matrix with labels');
    ax.set_xlabel('Predicted odors')
    ax.set_ylabel('Actual odors');


    if data_flag == '5 odors' or data_flag == '5 odors shuffle':
        ax.xaxis.set_ticklabels(['Benz', 'Cit','Ethyl', 'Geran','Hex'])
        ax.yaxis.set_ticklabels(['Benz', 'Cit','Ethyl', 'Geran', 'Hex'])
    if data_flag == 'mix1' or data_flag == 'mix1 shuffle':
        ax.xaxis.set_ticklabels(['Cit', 'Benz','Mix1'])
        ax.yaxis.set_ticklabels(['Cit', 'Benz','Mix1'])
    if data_flag == 'mix2' or data_flag == 'mix2 shuffle':
        ax.xaxis.set_ticklabels(['Ethyl', 'Hex','Mix2'])
        ax.yaxis.set_ticklabels(['Ethyl', 'Hex','Mix2'])
    if data_flag == 'benz hex ethyl' or data_flag == 'benz hex ethyl shuffle':
        ax.xaxis.set_ticklabels(['Benz', 'Ethyl','Hex'])
        ax.yaxis.set_ticklabels(['Benz', 'Ethyl','Hex'])        
    if data_flag == 'cit ger lemon' or data_flag == 'cit ger lemon shuffle':
        ax.xaxis.set_ticklabels(['Cit', 'Geran','Lemon'])
        ax.yaxis.set_ticklabels(['Cit', 'Geran','Lemon'])
    if data_flag == 'concentrations':
        ax.xaxis.set_ticklabels(['10^1','10^0','10^-1', '10^-2','10^-3','10^-4'])
        ax.yaxis.set_ticklabels(['10^1','10^0','10^-1', '10^-2','10^-3','10^-4'])
    if data_flag == '3odor cons':
        ax.xaxis.set_ticklabels(['10^0', 'cit','10^-2', 'ethyl','10^-4','hex'])
        ax.yaxis.set_ticklabels(['10^0', 'cit','10^-2', 'ethyl','10^-4','hex'])
    if data_flag == '8':
        ax.xaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
        ax.yaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
    if data_flag == '8 odors shuffle':
        ax.xaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
        ax.yaxis.set_ticklabels(['Benz' , 'Hex' , 'Ethyl' , 'Rose' , 'Lem' , 'Ger' , 'Cit' , 'Van'])
    if data_flag == 'ros lemon':
        ax.xaxis.set_ticklabels(['Lemon', 'Rosmary'])
        ax.yaxis.set_ticklabels(['Lemon', 'Rosmary'])
    #plt.xlabel('Predicted')
    #plt.ylabel('Truth')
    #plt.title('Confusion Matrix of Cross Validation leave one group out')
    plt.show()
    Accuracy = np.mean(np.array(accuracy_all))
    Accuracy = str(round(Accuracy*100, 2))
    # Std = np.std(np.array(accuracy_all))
    # Std = str(round(Std, 3))
    print('The accuracy of the cross validation is: ' + str(Accuracy) + '% for ' + str(len(df.index)) + ' samples total')

# ------------ Cross validation considering IDs -------------------------------------
def cross_validation_ID(X_normalized,y,df,RandomForest):
    if str(type(X_normalized)) == "<class 'pandas.core.frame.DataFrame'>":
        X_normalized = X_normalized.values
    if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
        y = y.values
    groups = df['ID'].values
    # group_kfold = GroupKFold(n_splits=10)
    group_kfold = StratifiedGroupKFold(n_splits = 10)
    # predicted = cross_val_predict(RandomForest, X_normalized, y, cv=group_kfold,groups=groups) 
    # accuracy_score(y, predicted)
    # print(scores)
    # print(np.mean(scores))
    ypred_all = []
    # accuracy_all = []
    save_test_index = []
    ytrue = []
    for train_index, test_index in group_kfold.split(X_normalized,y,groups = groups):
        save_test_index.append(test_index)
        RandomForest.fit(X_normalized[train_index], y[train_index])
        ypred = RandomForest.predict(X_normalized[test_index])
        ytrue.append(y[test_index])
        # print ("y pred: " ,ypred)
        # pring ("y true" , ytrue)
        # accur = accuracy_score(y[test_index], ypred)
        # print("predicted labels for data of indices", test_index, "are:", ypred)
        ypred_all.append(ypred)
        # accuracy_all.append(accur)
   
    # find the nearest fold from the average of all folds
    # index_mean_accuracy = accuracy_all.index(np.array(accuracy_all).flat[np.abs(accuracy_all - sum(accuracy_all)/len(accuracy_all)).argmin()])
    # ypred_mean = ypred_all[index_mean_accuracy]
    # ypred_mean_index = save_test_index[index_mean_accuracy]
    
    flat_list_pred = [item for sublist in ypred_all for item in sublist]
    ytrue = [item for sublist in ytrue for item in sublist]
    accuracy_all = accuracy_score(ytrue, flat_list_pred)
    # return ypred_mean,ypred_mean_index,accuracy_all
    return flat_list_pred,accuracy_all,ytrue
# ------------ cross validation not considering ID-----------------------------------

def cross_validation_noID(X_normalized,y,RandomForest):
    if str(type(X_normalized)) == "<class 'pandas.core.frame.DataFrame'>":
        X_normalized = X_normalized.values
    if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
        y = y.values
    
    kfold = KFold(n_splits=10,shuffle = True)
    ypred_all = []
    accuracy_all = []
    save_test_index = []
    for train_index, test_index in kfold.split(X_normalized,y):
        save_test_index.append(test_index)
        RandomForest.fit(X_normalized[train_index], y[train_index])
        ypred = RandomForest.predict(X_normalized[test_index])
        accur = accuracy_score(y[test_index], ypred)
        # print("predicted labels for data of indices", test_index, "are:", ypred)
        ypred_all.append(ypred)
        accuracy_all.append(accur)
      
    # find the nearest fold from the average of all folds
    index_mean_accuracy = accuracy_all.index(np.array(accuracy_all).flat[np.abs(accuracy_all - sum(accuracy_all)/len(accuracy_all)).argmin()])
    ypred_mean = ypred_all[index_mean_accuracy]
    ypred_mean_index = save_test_index[index_mean_accuracy]
    
    # RandomForest.fit(X_train_normalized,y_train)
    # scores = cross_val_score(RandomForest,X_normalized,y,cv=10)
    # print('The mean accuracy of 10 folds: ' + str(np.mean(scores)*100) + ' with standard deviation of ' + str(scores.std()))
    
    return ypred_mean,ypred_mean_index,accuracy_all

# ------------ Cross validation leave one group out (considering IDs) -------------------------------------
# def cross_validation_ID_LOGOCV(X_normalized,y,df,RandomForest):
#     if str(type(X_normalized)) == "<class 'pandas.core.frame.DataFrame'>":
#         X_normalized = X_normalized.values
#     if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
#         y = y.values
#     groups = df['ID'].values
#     group_LOO = LeaveOneGroupOut()
#     ypred_all = []
#     accuracy_all = []
#     save_test_index = []
#     for train_index, test_index in group_LOO.split(X_normalized,y,groups = groups):
#         save_test_index.append(test_index)
#         RandomForest.fit(X_normalized[train_index], y[train_index])
#         ypred = RandomForest.predict(X_normalized[test_index])
#         accur = accuracy_score(y[test_index], ypred)
#         print("predicted labels for data of indices", test_index, "are:", ypred)
#         ypred_all.append(ypred)
#         accuracy_all.append(accur)
   
#     # find the nearest fold from the average of all folds
#     index_mean_accuracy = accuracy_all.index(np.array(accuracy_all).flat[np.abs(accuracy_all - sum(accuracy_all)/len(accuracy_all)).argmin()])
#     ypred_mean = ypred_all[index_mean_accuracy]
#     ypred_mean_index = save_test_index[index_mean_accuracy]
    
#     return ypred_mean,ypred_mean_index,accuracy_all

def cross_validation_ID_LOGOCV(X_normalized,y,df,RandomForest):
    if str(type(X_normalized)) == "<class 'pandas.core.frame.DataFrame'>":
        X_normalized = X_normalized.values
    if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
        y = y.values
    groups = df['ID'].values
    group_LOO = LeaveOneGroupOut()
    ypred_all = []
    # accuracy_all = []
    # save_test_index = []
    ytrue = []
    for train_index, test_index in group_LOO.split(X_normalized,y,groups = groups):
        # save_test_index.append(test_index)
        RandomForest.fit(X_normalized[train_index], y[train_index])
        ypred = RandomForest.predict(X_normalized[test_index])
        ytrue.append(y[test_index])
        # accur = accuracy_score(y[test_index], ypred)
        # print("predicted labels for data of indices", test_index, "are:", ypred) 
        ypred_all.append(ypred)
        # accuracy_all.append(accur)
   
    # # find the nearest fold from the average of all folds
    # index_mean_accuracy = accuracy_all.index(np.array(accuracy_all).flat[np.abs(accuracy_all - sum(accuracy_all)/len(accuracy_all)).argmin()])
    # ypred_mean = ypred_all[index_mean_accuracy]
    # ypred_mean_index = save_test_index[index_mean_accuracy]
    flat_list_pred = [item for sublist in ypred_all for item in sublist]
    ytrue = [item for sublist in ytrue for item in sublist]
    accuracy_all = accuracy_score(ytrue, flat_list_pred)
    # return ypred_mean,ypred_mean_index,accuracy_all
    return flat_list_pred,accuracy_all,ytrue
    # return ypred_mean,ypred_mean_index,accuracy_all


def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list