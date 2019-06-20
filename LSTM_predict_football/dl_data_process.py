# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:10:23 2018

@author: Administrator


深度学习赔率数据

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
import tfb_main 

import numpy as np

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

TIME_STEPS = 31     # same as the height of the image
INPUT_SIZE = 6     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 3
CELL_SIZE = 50
LR = 0.001





#1
rs0='/tfbDat/'
fgid,fxdat=rs0+'gid2019_js.dat', rs0+'xdat2017.dat'


def kelly_variance(prob0, prob9, odds_oz0, odds_oz9, vback0, vback9):
    odds_oz0 = odds_oz0.values
    odds_oz9 = odds_oz9.values
    prob0 = prob0.values
    prob9 = prob9.values
    vback0 = vback0.values
    vback9 = vback9.values
    
    
    prob0_mean = np.mean(prob0, axis=0)
    prob9_mean = np.mean(prob9, axis=0)

    kelly0 = odds_oz0 * prob0_mean / 100
    kelly9 = odds_oz9 * prob9_mean / 100
    
    kelly0_mean = np.mean(kelly0, axis=0)
    kelly9_mean = np.mean(kelly9, axis=0)
    
    kelly0_var = (kelly0-kelly0_mean) * (kelly0-kelly0_mean) *100
    kelly9_var = (kelly9-kelly9_mean) * (kelly9-kelly9_mean) *100
    
    ky0_m = np.mean(kelly0_var, axis=0)
    ky9_m = np.mean(kelly9_var, axis=0)
    
    ky0_var = np.vstack([ky0_m, ky0_m, ky0_m])
    ky9_var = np.vstack([ky9_m, ky9_m, ky9_m])
    
    kelly0_diff = kelly0 / vback0
    kelly9_diff = kelly9 / vback9
    
    kelly0 = pd.DataFrame(kelly0)
    kelly9 = pd.DataFrame(kelly9)
    kelly0_var = pd.DataFrame(kelly0_var)
    kelly9_var = pd.DataFrame(kelly9_var)
    kelly0_diff = pd.DataFrame(kelly0_diff)
    kelly9_diff = pd.DataFrame(kelly9_diff)
    ky0_var = pd.DataFrame(ky0_var)
    ky9_var = pd.DataFrame(ky9_var)
    
    return kelly0, kelly9,  kelly0_var, kelly9_var, kelly0_diff, kelly9_diff, ky0_var,ky9_var
    


def save_odds_to_file_for_svm(file_name, num):
    #2---init.tfb
    xtfb = tft.fb_init(rs0, fgid)  
    df =  tfsys.gids 
    p_data = pd.DataFrame()
    p_data_oz = pd.DataFrame()
    p_data_az = pd.DataFrame()
    oz_cols = ['pwin0','pdraw0','plost0',
            'pwin9','pdraw9','plost9',
            'vwin0','vdraw0','vlost0','vback0','vwin0kali','vdraw0kali','vlost0kali',
            'vwin9','vdraw9','vlost9','vback9','vwin9kali','vdraw9kali','vlost9kali']
    az_cols = ['mshui0', 'pan0', 'gshui0','mshui9', 'pan9', 'gshui9']
    
    kali0 = ['vwin0kali','vdraw0kali','vlost0kali']
    kali9 = ['vwin9kali','vdraw9kali','vlost9kali']    
    oz0 = ['pwin0','pdraw0','plost0']
    oz9 = ['pwin9','pdraw9','plost9']
    az0 = ['mshui0', 'pan0', 'gshui0']
    az9 = ['mshui9', 'pan9', 'gshui9']
    pb0 = ['vwin0','vdraw0','vlost0']
    pb9 = ['vwin9','vdraw9','vlost9']
    vb0 =['vback0']
    vb9 =['vback9']
    
    oz = ['pwin0','pwin9','pdraw0','pdraw9','plost0','plost9']
    
    kali_columns = ['vwinkali','vdrawkali','vlostkali']
    oz_columns = ['pwin','pdraw','plost']
    az_columns = ['mshui', 'gshui', 'pan'] 

    
    for i, row in df.iterrows():
        if ((i+1) % 2000 == 0):
            print((i+1)/len(df) * 100, "%")
            print('now:',zt.tim_now_str())
        if i>=num[0] and i<num[1]:    
            gid = row['gid']
            fxdat_oz = tfsys.rxdat + gid + '_oz.dat'
            fxdat_az = tfsys.rxdat + gid + '_az.dat'
            
            if os.path.exists(fxdat_oz) and os.path.exists(fxdat_az):
                odds_oz = pd.read_csv(fxdat_oz, index_col = False, dtype = str, encoding = 'gb18030')  
                odds_az = pd.read_csv(fxdat_az, index_col = False, dtype = str, encoding = 'gb18030')  
                
                if len(odds_oz) >= tfsys.cidrows and odds_oz.loc[0, 'kwin'] != '-1' and len(odds_az) >= tfsys.cidrows and odds_az.loc[0, 'kwin'] != '-1':    #如果数据有CID_ROWS行并且有比赛结果才处理数据
                    if float(odds_oz.loc[0, 'pwin0']) <= 1.5 or float(odds_oz.loc[0, 'plost0']) <= 1.5  or float(odds_oz.loc[0, 'pwin9']) <= 1.5 or float(odds_oz.loc[0, 'plost9']) <= 1.5  :
                           continue
                    cids = ['2','3','293']
                    odds_oz = odds_oz[odds_oz['cid'].isin(cids)]
                    if len(odds_oz)!=3: continue
#                    odds_oz = odds_oz[1:tfsys.cidrows+1]
#                    odds_az = odds_az[0:tfsys.cidrows]
                    target = odds_oz.pop('kwin') 
                    odds_oz = odds_oz[oz_cols].astype(float)
#                    odds_az = odds_az[az_cols].astype(float) 
                    #1
                    odds_oz0_9 = odds_oz[oz]
                    odds_oz0 = odds_oz[oz0]
                    odds_oz9 = odds_oz[oz9]
#                    odds_az0 = odds_az[az0]
#                    odds_az9 = odds_az[az9]
                    odds_oz0.columns = oz_columns
                    odds_oz9.columns = oz_columns
#                    odds_az0.columns = az_columns
#                    odds_az9.columns = az_columns                
#                    az_diff = odds_az9 - odds_az0
                    oz_diff = odds_oz9 / odds_oz0
                    #2
                    odds_kali0 = odds_oz[kali0]
                    odds_kali9 = odds_oz[kali9]
                    odds_kali0.columns = kali_columns
                    odds_kali9.columns = kali_columns
                    kali_diff = odds_kali9 - odds_kali0 
                
                    #3
                    vback0 = odds_oz['vback0'] /100
                    vback9 = odds_oz['vback0'] /100                   
                    vback0_diff = odds_kali0.sub(vback0,axis=0)
                    vback9_diff = odds_kali9.sub(vback9,axis=0)
                    
                    #4
                    prob0 = odds_oz[pb0]
                    prob9 = odds_oz[pb9]
                   
                    
                    target = target.astype(int)
                    target[target==3] = 2
                    
                    target = target.reset_index(drop=True)
#                    az_diff = az_diff.reset_index(drop=True)
                    oz_diff = oz_diff.reset_index(drop=True)
                    kali_diff = kali_diff.reset_index(drop=True)
                    vback0_diff = vback0_diff.reset_index(drop=True)
                    vback9_diff = vback9_diff.reset_index(drop=True)

#                    odds_az0 = odds_az0.reset_index(drop=True)
                    odds_oz0 = odds_oz0.reset_index(drop=True) 
#                    odds_az9 = odds_az9.reset_index(drop=True)
                    odds_oz9 = odds_oz9.reset_index(drop=True)
                    odds_oz0_9 = odds_oz0_9.reset_index(drop=True)
                    odds_kali0 = odds_kali0.reset_index(drop=True)
                    odds_kali9 = odds_kali9.reset_index(drop=True)
                    vback0 = vback0.reset_index(drop=True)
                    vback9 = vback9.reset_index(drop=True)
                    prob0 = prob0.reset_index(drop=True)
                    prob9 = prob9.reset_index(drop=True)
                    
                    kelly0, kelly9,  kelly0_var, kelly9_var, kelly0_diff, kelly9_diff, ky0_var,ky9_var = kelly_variance(prob0, prob9, odds_oz0, odds_oz9, vback0, vback9)
                    
                    
#                    merge_data = pd.concat([target, az_diff, oz_diff, kali_diff, vback0_diff, vback9_diff], axis=1)                                                     
                    merge_data = pd.concat([target, odds_oz0_9, ky0_var, ky9_var], axis=1)                 
                    p_data = p_data.append(merge_data, ignore_index=True)

    p_data.to_csv(file_name, index=False, encoding='gb18030')












###########################################################################################
###########################################################################################

'''
读入赔率文件
'''
def calculate_odds_cov(odds_oz):
    win0 =  odds_oz['win0']
    draw0 = odds_oz['draw0']
    lost0 = odds_oz['lost0']    

    win9 =  odds_oz['win9']
    draw9 = odds_oz['draw9']
    lost9 = odds_oz['lost9']  
    
    cov_win0 = win0.cov(draw0)
    cov_lost0 = lost0.cov(draw0)
    cov_win9 = win9.cov(draw9)
    cov_lost9 = lost9.cov(draw9)
       
    return cov_win0, cov_lost0, cov_win9, cov_lost9


def calculate_odds_cv(odds_oz):
#    odds_oz = odds_oz.astype(float)
#    win9_0 = odds_oz['win9'] - odds_oz['win0']
#    draw9_0 = odds_oz['draw9'] - odds_oz['draw0']
#    lost9_0 = odds_oz['lost9'] - odds_oz['lost0']
    
    win_cv_0 = odds_oz['win0'].std() / odds_oz['win0'].mean()
    draw_cv_0 = odds_oz['draw0'].std() / odds_oz['draw0'].mean()
    lost_cv_0 = odds_oz['lost0'].std() / odds_oz['lost0'].mean()
    
    win_cv_9 = odds_oz['win9'].std() / odds_oz['win9'].mean()
    draw_cv_9 = odds_oz['draw9'].std() / odds_oz['draw9'].mean()
    lost_cv_9 = odds_oz['lost9'].std() / odds_oz['lost9'].mean()
    win_cv_0 = round(win_cv_0*100, 3)
    draw_cv_0 = round(draw_cv_0*100, 3)
    lost_cv_0 = round(lost_cv_0*100, 3)
    win_cv_9 = round(win_cv_9*100, 3)
    draw_cv_9 = round(draw_cv_9*100, 3)
    lost_cv_9 = round(lost_cv_9*100, 3) 

    return win_cv_0, draw_cv_0, lost_cv_0, win_cv_9, draw_cv_9, lost_cv_9

#### 集成用于xgboost训练的样本数据 #####
def save_odds_to_file_for_xgboost(file_name, num):
    cv_sgn = ['win_cv_0', 'draw_cv_0', 'lost_cv_0', 'win_cv_9', 'draw_cv_9', 'lost_cv_9']
    #2---init.tfb
    rs0='/tfbDat/'
    fgid=rs0+'gid2018-2019(xgboost).dat'
    xtfb = tft.fb_init(rs0, fgid)  
    df =  tfsys.gids  
    p_data = pd.DataFrame()
    p_data_tz = pd.DataFrame()
    oz_cv = pd.DataFrame()
    
    for i, row in df.iterrows():
        if ((i+1) % 2000 == 0):
            print((i+1)/len(df) * 100, "%")
            print('now:',zt.tim_now_str())
        if i>=num[0] and i<num[1]: 
            kend = row['kend']
            kwin = row['kwin']
            if kend == '0' or kwin == -1 : continue
        
            gid = row['gid']
            fxdat_tz = tfsys.rxdat + gid + '_tz.dat'
            fxdat_oz = tfsys.rxdat + gid + '_oz_1.dat'
            
            if os.path.exists(fxdat_tz) and os.path.exists(fxdat_oz):
                odds_tz = pd.read_csv(fxdat_tz, index_col = False, dtype = float, encoding = 'gb18030')  
                odds_oz = pd.read_csv(fxdat_oz, index_col = False, dtype = float, encoding = 'gb18030')  
                if odds_tz.isnull().values.any(): continue  ##如果有Nan值就不保存该数据 
               
                win_cv_0, draw_cv_0, lost_cv_0, win_cv_9, draw_cv_9, lost_cv_9 = calculate_odds_cv(odds_oz)
#                cov_win0, cov_lost0, cov_win9, cov_lost9 = calculate_odds_cov(odds_oz)
                features = odds_tz
                features['win_cv_0'] =  win_cv_0
                features['draw_cv_0'] = draw_cv_0
                features['lost_cv_0'] = lost_cv_0
                features['win_cv_9'] = win_cv_9
                features['draw_cv_9'] = draw_cv_9
                features['lost_cv_9'] = lost_cv_9
                
                p_data = p_data.append(features, ignore_index=True)
                   
    p_data.to_csv(file_name, index=False, encoding='gb18030')

  




def one_hot(labels, n_class = 3):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y




########################################
# 特征：亏损与抽水之间的比值
def loss_feature(data):
    volume = data[['volume_h', 'volume_d', 'volume_g']]
    profit = data[['profit_h', 'profit_d', 'profit_g']]
    idx = np.argmin(profit.values, axis=1)
    loss = profit.min(axis=1)
    mask = copy.deepcopy(profit) 
    mask[mask>loss[0]] = 1
    mask[mask==loss[0]] = 0
    mask.columns = ['volume_g', 'volume_d', 'volume_m']
    vol_sum = volume.sum(axis=1)  
    back_vol = vol_sum * (1-data['bf_back']/100)
    loss_ratio = abs(loss) / back_vol
    data['idx'] = idx
    data['loss_rio'] = loss_ratio    
    
    return idx, loss_ratio
      
##########################################
# 特征：成交量与胜率的比值
def volume_feature(data):
    volume = data[['volume_h', 'volume_d', 'volume_g']]
    prob = data[['av_prob_h', 'av_prob_d', 'av_prob_g']]
    
    vol_prob = volume / prob
    
    return vol_prob


##################################################
    


##################################################



def load_odds_file_xgboost(file):
    features = pd.DataFrame()
    data = pd.read_csv(file)
    target = data.pop('kwin') ##目标分类
    target = target.astype(int)
    target[target==1] = 0
    target[target==3] = 1
    
    qj = data.pop('qj')
    qs = data.pop('qs')
    
    
#    x = (data['av_odd_h']*data['av_odd_h'] + data['av_odd_g']*data['av_odd_g'] - data['av_odd_d']*data['av_odd_d']) / 2*data['av_odd_h']
#    yy = data['av_odd_h']*data['av_odd_h'] - x*x
#    point = yy>0
#    y = math.sqrt(abs(yy))
    
#    rio = x/y
#    point = rio>
    av_odd_h = data['av_odd_h']
    av_odd_d = data['av_odd_d']
    av_odd_g = data['av_odd_g']
    
    odd_ratio = ((av_odd_h-av_odd_d) * (av_odd_h-av_odd_d) - (av_odd_g-av_odd_d) * (av_odd_g-av_odd_d)) / (av_odd_d * av_odd_d)
    
#    rio = (data['av_odd_h'] - data['av_odd_g']) /  data['av_odd_d']
    data['odd_rio'] = odd_ratio
    
    shui = data['volume_h']+data['volume_d']+data['volume_g']
#    data['profit_h'] = data['profit_h'] / (shui*(1-data['av_back']/100))
#    data['profit_d'] = data['profit_d'] / (shui*(1-data['av_back']/100))
#    data['profit_g'] = data['profit_g'] / (shui*(1-data['av_back']/100))
    
#    data['av_odd_h'] = data['av_odd_h'] - data['bf_odd_h']
#    data['av_odd_d'] = data['av_odd_d'] - data['bf_odd_d']
#    data['av_odd_g'] = data['av_odd_g'] - data['bf_odd_g']
    
#    data['av_prob_h'] = data['av_prob_h'] - data['bf_prob_h']
#    data['av_prob_d'] = data['av_prob_d'] - data['bf_prob_d']
#    data['av_prob_g'] = data['av_prob_g'] - data['bf_prob_g']
 
    #####

    idx, loss_ratio = loss_feature(data)
    vol_prob = volume_feature(data)
    idx = keras.utils.to_categorical(idx, 3)
    
#    features['odd_ratio'] = odd_ratio
#    features['idx'] = idx
#    features['idx0'] = idx[:, 0]
#    features['idx1'] = idx[:, 1]
#    features['idx2'] = idx[:, 2]
#    features['loss_rio'] = loss_ratio
#    features['av_back'] = data['av_back']

    features['av_odd_h'] = data['av_odd_h']
    features['av_odd_d'] = data['av_odd_d']
    features['av_odd_g'] = data['av_odd_g']
    
    features['win_cv_0'] = data['win_cv_0']
    features['draw_cv_0'] = data['draw_cv_0']
    features['lost_cv_0'] = data['lost_cv_0']
    features['win_cv_9'] = data['win_cv_9']
    features['draw_cv_9'] = data['draw_cv_9']
    features['lost_cv_9'] = data['lost_cv_9']

    features['shui_h'] = data['shui_h']
    features['pan'] = data['pan']
    features['shui_g'] = data['shui_g']
    
    features['hot_idx_h'] = data['hot_idx_h']
    features['hot_idx_d'] = data['hot_idx_d']
    features['hot_idx_g'] = data['hot_idx_g']

    features['pf_idx_h'] = data['pf_idx_h']
    features['pf_idx_d'] = data['pf_idx_d']
    features['pf_idx_g'] = data['pf_idx_g']
    
    
    
#    data.pop('profit_h')
#    data.pop('profit_d')
#     data.pop('profit_g')
            
    features = features.astype(float)
    
    features = np.array(features)
    target = np.array(target)
    
    return features, target 




def load_data(file):
    target, dataset = load_odds_file(file)

    # split into train and test sets
#    train_size = int(len(dataset) * 0.7)
#    test_size = len(dataset) - train_size
#    x_train, x_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#    y_train, y_test = target[0:train_size], target[train_size:len(dataset)]


    x_train, x_test, y_train, y_test = train_test_split(dataset, target, train_size=0.70, random_state=0)
    
    return (x_train, y_train), (x_test, y_test) 





        

      







def load_data_xgboost(file):
    data, target = load_odds_file_xgboost(file)
    
    return data, target
