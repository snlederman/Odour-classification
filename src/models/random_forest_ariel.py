# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:10:24 2021

@author: ariel
"""
# %%
# Import libraries
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import HelpFunctions as HF

# %%-----5 odors----------------------------------------------------------------------------------------------------------------------------------
# Reading csv file
#df_neta = pd.read_csv('8_odors_new_names.csv')
print('Enter the desired data to analyse (5 odors,mix1,mix2,benz hex ethyl, cit ger,concentrations, 3odor cons, 8 odors, vanilla ros lemon)')
data_flag = input()
if data_flag == '8':
     df = pd.read_csv('8_odors_new_names.csv')
if data_flag == 'mix1':
    df = pd.read_csv('Mix1_new.csv')
if data_flag == 'mix2':
  df = pd.read_csv('mix2_new.csv')
if data_flag == 'benz hex ethyl':
     df = pd.read_csv('benz_hex_ethyl.csv')
if data_flag == 'cit ger lemon':
    df = pd.read_csv('cit_ger_lemon_new.csv')
if data_flag == 'Lem Ros':
     df = pd.read_csv('Lemon_Rosemary.csv')
#
# if data_flag == '5 odors shuffle':
#     df = pd.read_csv('normalized/5_odors_shuffled.csv')
# if data_flag == 'mix1 shuffle':
#     df = pd.read_csv('normalized/mix1_shuffled.csv')
# if data_flag == 'mix2 shuffle':
#     df = pd.read_csv('normalized/mix2_shuffled.csv')
# if data_flag == 'benz hex ethyl shuffle':
#     df = pd.read_csv('normalized/benz_hex_ethyl_shuffled.csv')
# if data_flag == 'cit ger shuffle':
#     df = pd.read_csv('normalized/cit_ger_shuffled.csv')
#
if data_flag == 'concentrations':
    df = pd.read_csv('concentrations_new.csv')
# if data_flag == '3odor cons':
#     df = pd.read_csv('normalized/3odors_cons.csv')
# if data_flag == '8 odors':
#     df = pd.read_csv('normalized/8_odors_new_names.csv')
if data_flag == '8 odors shuffle':
    df = pd.read_csv('8_odors_new_names_shuffled.csv')
# if data_flag == 'ros lemon':
#     df = pd.read_csv('normalized/Ros_lemon_new.csv')
if data_flag == 'test_train 10^0':
     df = pd.read_csv('Benz_10^0_Train.csv')
# if data_flag == 'test_train 10^-1':
#     df = pd.read_csv('normalized/Benz_10^-1_Train.csv')
if data_flag == 'test_train 10^-2':
    df = pd.read_csv('Benz_10^-2_Train.csv')
#if data_flag == 'test_train 10^-3':
#     df = pd.read_csv('normalized/Benz_10^-3_Train.csv')
if data_flag == 'test_train 10^-4':
     df = pd.read_csv('Benz_10^-4_Train.csv')
# if data_flag == 'test_train 10^1':
#     df = pd.read_csv('normalized/Benz_10^1_Train.csv')
# %%  split to train and test, choosing whether or not consider ID of antenna using flag = 1 as considering
# Test set 20% and training set 80%
print('Enter 1 if you want to split the dataset once, else enter 0 (for cross validation)')
flag_split = int(input())
# flag_split = 0
print('Enter 1 if you want to consider ID of the antenna in the split and 0 otherwise')
flag_ID = int(input())
# flag_ID = 1
if flag_split:
    if not data_flag == 'test_train 10^0' and not data_flag == 'test_train 10^-1' and not data_flag == 'test_train 10^-2' and not data_flag == 'test_train 10^-3' and not data_flag == 'test_train 10^-4' and not data_flag == 'test_train 10^1':
        X, y, X_train, X_test, y_train, y_test = HF.split_once(df, flag_ID)
    else:
        X = df.drop(['label', 'ID', 'channel', 'date'], axis='columns')
        y = df['label']
        X_train = df.drop(['label', 'ID', 'date', 'channel'], axis='columns')
        y_train = df['label']
        if data_flag == 'test_train 10^0':
            test = pd.read_csv('Benz_10^0_Test.csv')
        if data_flag == 'test_train 10^-1':
            test = pd.read_csv('Benz_10^-1_Test.csv')
        if data_flag == 'test_train 10^-2':
            test = pd.read_csv('Benz_10^-2_Test.csv')
        if data_flag == 'test_train 10^-3':
            test = pd.read_csv('Benz_10^-3_Test.csv')
        if data_flag == 'test_train 10^-4':
            test = pd.read_csv('Benz_10^-4_Test.csv')
        if data_flag == 'test_train 10^1':
            test = pd.read_csv('Benz_10^1_Test.csv')
        X_test = test.drop(['label', 'ID', 'date', 'channel'], axis='columns')
        y_test = test['label']
else:
    X = df.drop(['label', 'ID', 'channel', 'date'], axis='columns')
    y = df['label']

# %% Normalization
# print('Enter 1 for normalize to 1 peak and 0 for normalize between -1 to 1')
# norm_flag = int(input())
norm_flag = 1
# dividing by the minimum value of each row (old normalization)
if norm_flag:
    X_normalized = X.div(X.min(axis=1), axis=0)
    if flag_split:
        X_train_normalized = X_train.div(X_train.min(axis=1), axis=0)
        X_test_normalized = X_test.div(X_test.min(axis=1), axis=0)

# Normalization by (-1,1)
if not norm_flag:
    X_normalized = X.apply(lambda x: (2 * (x - x.min()) / (x.max() - x.min())) - 1, axis=1)
    if flag_split:
        X_train_normalized = X_train.apply(lambda x: (2 * (x - x.min()) / (x.max() - x.min())) - 1, axis=1)
        X_test_normalized = X_test.apply(lambda x: (2 * (x - x.min()) / (x.max() - x.min())) - 1, axis=1)

# plot data of normalization (to check)
# HF.plot_check_normalization(X_normalized)

# %% Classification using random forest:
RandomForest = RandomForestClassifier(n_estimators=100)

# %% cross validation
# by groups (of IDs)
if flag_ID and not flag_split:
    # print('Leave one out? (write 1 if yes and 0 if no)')
    # leaveOneOut = int(input())
    leaveOneOut = 1
    if leaveOneOut:
        ypred_all_cv, accuracy_all, ytrue = HF.cross_validation_ID_LOGOCV(X_normalized, y, df, RandomForest)
    else:
        ypred_all_cv, accuracy_all, ytrue = HF.cross_validation_ID(X_normalized, y, df, RandomForest)

# not considering ID
if not flag_ID and not flag_split:
    ypred_mean, ypred_mean_index, accuracy_all = HF.cross_validation_noID(X_normalized, ytrue, RandomForest)

# %% visualize confusion matrix
if flag_split:
    HF.confusion_mat_graph_onesplit(RandomForest, X_train, y_train, X_test, y_test, data_flag, df)
else:
    HF.confusion_mat_graph_cv(ytrue, ypred_all_cv, accuracy_all, df, data_flag)

# %% PCA
# pca = PCA(n_components=15)
# pca.fit(X)
# transformed_data_15_features = pca.transform(X_normalized)
# #%% cross validation by groups (of IDs) - PCA (no normalization,considerind IDs)
# ypred_all_cv,accuracy_all,ytrue = HF.cross_validation_ID(transformed_data_15_features,y,df,RandomForest)
# HF.confusion_mat_graph_cv(ytrue,ypred_all_cv,accuracy_all,df,data_flag)
# if str(type(transformed_data_15_features)) == "<class 'pandas.core.frame.DataFrame'>":
#     transformed_data_15_features = transformed_data_15_features.values
# if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
#     y = y.values
# groups = df['ID'].values
# group_kfold = GroupKFold(n_splits=10)

# ypred_all = []
# accuracy_all = []
# save_test_index = []
# for train_index, test_index in group_kfold.split(transformed_data_15_features,y,groups = groups):
#     save_test_index.append(test_index)
#     RandomForest.fit(transformed_data_15_features[train_index], y[train_index])
#     ypred = RandomForest.predict(transformed_data_15_features[test_index])
#     accur = accuracy_score(y[test_index], ypred)
#     # print("predicted labels for data of indices", test_index, "are:", ypred)
#     ypred_all.append(ypred)
#     accuracy_all.append(accur)

# index_max_accuracy = accuracy_all.index(max(accuracy_all))
# ypred_best = ypred_all[index_max_accuracy]
# ypred_best_index = save_test_index[index_max_accuracy]
