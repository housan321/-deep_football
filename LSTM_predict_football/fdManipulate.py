# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 23:16:31 2019

@author: ASUS
"""
import os,sys,re
import arrow,bs4
import pandas as pd

import requests
from bs4 import BeautifulSoup 

import copy
import math
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
import zpd_talib as zta
#
import tfb_sys as tfsys
import tfb_tools as tft
import tfb_strategy as tfsty

import numpy as np

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



def read_lstm_sample_files(file_name, num):
    #2---init.tfb
    rs0='/tfbDat/'
    fgid=rs0+'gid2019_js.dat'
    xtfb = tft.fb_init(rs0, fgid)  
    df =  tfsys.gids  
    data = pd.DataFrame()
    
    for i, row in df.iterrows():
        if ((i+1) % 2000 == 0):
            print((i+1)/len(df) * 100, "%")
            print('now:',zt.tim_now_str())
        if i>=num[0] and i<num[1]: 
            kend = row['kend']
            kwin = row['kwin']
            if kend == '0' or kwin == -1 : continue
        
            gid = row['gid']
            fxdat_ftr = tfsys.rxdat + gid + '_ftr.dat'
            
            if os.path.exists(fxdat_ftr):
                features = pd.read_csv(fxdat_ftr, index_col = False,  encoding = 'gb18030')  
                if features.isnull().values.any(): continue  ##如果有Nan值就不保存该数据 
                
                data = data.append(features, ignore_index=True)
    data.loc[:,"FTR"][data.loc[:,"FTR"]==1] = 0   
    data.loc[:,"FTR"][data.loc[:,"FTR"]==3] = 1 
  
    data=data.drop(['vol_prob_h', 'vol_prob_d','vol_prob_g', 'loss_idx', 'loss_vol_ratio'],axis=1)
    data=data.drop(['avg_win', 'avg_draw','avg_lost','avg_win_diff', 'avg_draw_diff','avg_lost_diff'],axis=1)

    return data




lstm_sample_files = 'data/lstm_samples.dat'
data = read_lstm_sample_files(lstm_sample_files, [0, 3000])


data.head()

ftrLE=LabelEncoder()
data.FTR=ftrLE.fit_transform(data.FTR)

hmLE=LabelEncoder()
data.HM1=hmLE.fit_transform(data.HM1)
data.head()

data.HM2=hmLE.fit_transform(data.HM2)
data.HM3=hmLE.fit_transform(data.HM3)
data.HM4=hmLE.fit_transform(data.HM4)
data.HM5=hmLE.fit_transform(data.HM5)
data.AM1=hmLE.fit_transform(data.AM1)
data.AM2=hmLE.fit_transform(data.AM2)
data.AM3=hmLE.fit_transform(data.AM3)
data.AM4=hmLE.fit_transform(data.AM4)
data.AM5=hmLE.fit_transform(data.AM5)

data.VM1=hmLE.fit_transform(data.VM1)
data.VM2=hmLE.fit_transform(data.VM2)
data.VM3=hmLE.fit_transform(data.VM3)
data.VM4=hmLE.fit_transform(data.VM4)
data.VM5=hmLE.fit_transform(data.VM5)

# data.loss_idx=hmLE.fit_transform(data.loss_idx)


data.head()


onehotencoder=OneHotEncoder()
onehotencoder.fit(data.FTR.reshape(-1,1))

final=onehotencoder.transform([[each] for each in data.FTR]).toarray()

final.shape

data.loc[:,"final1"]=final[:,0]
data.loc[:,"final2"]=final[:,1]

dataT = data[1700:]
data = data[:1700]


data.to_csv('samples/allAtt_onehot_large_train.csv',index=None)
dataT.to_csv('samples/allAtt_onehot_large_test.csv',index=None)




