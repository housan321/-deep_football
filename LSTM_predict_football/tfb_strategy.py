# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 

  
文件名:tfb_strategy.py
默认缩写：import tfb_strategy as tfsty
简介：Top极宽量化·常用足彩策略模块
 

'''
#

import sys,os,re
import os,sys,re
import arrow,bs4,random

import numpy as np
import pandas as pd
import tushare as ts
#import talib as ta

import matplotlib as mpl
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
#import multiprocessing
#
#import arrow
import datetime as dt
import time
from dateutil.rrule import *
from dateutil.parser import *
import calendar as cal
#
import csv
import pickle
import numexpr as ne  

#
import requests
import bs4
from bs4 import BeautifulSoup 
from robobrowser import RoboBrowser 
#from selenium import webdriver

#
import zsys
import ztools as zt
import ztop_ai as zai
import zpd_talib as zta
#
#
import tfb_sys as tfsys
import tfb_tools as tft

import keras
from keras.models import Sequential
import os
from keras.models import load_model


# Imports for LSTM

from lstm_utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf


from tensorflow.contrib import rnn

load_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'cnn_model(7.10).h5' #第一代CNN模型
model_name1 = 'cnn_model(7.10).h5'
model_name2 = 'cnn_model(8.17_cnn(1.5)).h5'
#model_name2 = 'cnn_model(8.17_cnn(1.5)).h5 ' 
#model_name2 = 'cnn_model(kelly).h5'
# load model and weights
if not os.path.isdir(load_dir):
    os.makedirs(load_dir)
model_path1 = os.path.join(load_dir, model_name1)
model_path2 = os.path.join(load_dir, model_name2)
 


n_classes = 2
batch_size = 1
chunk_size=27
n_chunks=1
rnn_size=64

       
#----------sta.xxx



def recurrent_neural_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    x=tf.transpose(x,[1,0,2])
    print("transpose",x)
    x=tf.reshape(x,[-1,chunk_size])
    print("reshape",x)
    x=tf.split(x,n_chunks)
    print("split",x)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    


    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def sta_lstm_pre(xtfb, match_name):

    samples = xtfb.xdat10
    
    home_win = np.zeros((len(samples), 2), dtype=np.float32)
    home_win[:,0] = 1
    
    graph = tf.Graph()
    
    with graph.as_default():
        x = tf.placeholder('float', [None, n_chunks,chunk_size])    
        y = tf.placeholder('float')  
        
        prediction = recurrent_neural_model(x)
        Y_pred = tf.argmax(prediction, 1)
#        Y_pred = tf.nn.softmax(logits=prediction)
        
        saver = tf.train.Saver() 
    
    with tf.Session(graph=graph) as sess:
    # load the meta graph and weights
        saver.restore(sess, tf.train.latest_checkpoint('model'))
        samples=samples.astype(float)
        samples = samples.values
        samples = samples.reshape((samples.shape[0], n_chunks, chunk_size))
   
        ypred = sess.run(prediction, feed_dict={x: samples})
#        ypred = sess.run(Y_pred, feed_dict={x: samples})
        
        ypred = sess.run(tf.equal(tf.argmax(ypred, 1), tf.argmax(home_win, 1)))
        
        
        ypred = ypred.reshape(ypred.shape[0], 1)
              
    ret = np.concatenate((match_name.values, ypred), axis=1)
    ret = pd.DataFrame(ret, columns=['gset','mplay','gplay', '预测(True:主队)'])
    
#    if os.path.exists(xtfb.ktimStr+ ' CNN-predict.csv'):
#        old_ret = pd.read_csv(xtfb.ktimStr+ ' CNN-predict.csv', index_col=False, encoding='gb18030')
#        ret = pd.concat([old_ret, ret], axis=1)
    ret.to_csv(xtfb.ktimStr+ ' LSTM-predict.csv',  index=False, encoding='gb18030')  





#------------- sta01..sta    
def sta00_pre(xtfb):
    #
    return -9

def sta00_sta(xtfb,df):          
    
    #
    return -9
    
#------------- sta01..sta        
def sta01_sta(xtfb,df):          
    xkwin,k0=-9,xtfb.staVars[0]
    #---k0=1.1,k1=80
    df2=df[df.cid==xtfb.kcid]
    if len(df2.index)>0:
        dwin,dlose=df2['pwin0'][0],df2['plost0'][0]
        if dwin<=k0:xkwin=3
        elif dlose<=k0:xkwin=0
    #
    return xkwin    

 
def sta01ext_sta(xtfb,df):          
    xkwin,k30,k00=-9,xtfb.staVars[0] ,xtfb.staVars[1]
    #---k0=1.1,k1=80
    df2=df[df.cid==xtfb.kcid]
    if len(df2.index)>0:
        dwin,dlose=df2['pwin0'][0],df2['plost0'][0]
        if dwin<=k30:xkwin=3
        elif dlose<=k00:xkwin=0
    #
    return xkwin   
    
    
def sta10_sta(xtfb,df):          
    xkwin,k0,k1=-9,xtfb.staVars[0],xtfb.staVars[1]
    #---k0=1.1,k1=80
    df3=df[df.pwin0<k0]
    df0=df[df.plost0<k0]
    xn9=len(df.index)
    if xn9>0:
        kn3,kn0=len(df3.index)/xn9*100,len(df0.index)/xn9*100
        if kn3>k1:xkwin=3
        elif kn0>k1:xkwin=0
    #
    return xkwin    
    
#------------- sta30.mul.sta    
def sta310_pre(xtfb):
    df=xtfb.xdat10
    #
    df['kpwin']=round(df['pwin9']/df['pwin0']*100)
    df['kplost']=round(df['plost9']/df['plost0']*100)
    df['kpdraw']=round(df['pdraw9']/df['pdraw0']*100)
    #
    return df

def sta310_sta3(xtfb,df):
    df9=df[df.kpwin>100]
    df1=df[df.kpwin<=100]
    xn=len(df1.index)-len(df9.index)
    #
    xkwin,k0=-9,xtfb.staVars[0]
    if xn>k0:xkwin=3
    #
    return xkwin
    
def sta310_sta1(xtfb,df):
    df9=df[df.kpdraw>100]
    df1=df[df.kpdraw<=100]
    xn=len(df1.index)-len(df9.index)
    #
    xkwin,k0=-9,xtfb.staVars[0]
    if xn>k0:xkwin=1
    #
    return xkwin

def sta310_sta0(xtfb,df):
    df9=df[df.kplost>100]
    df1=df[df.kplost<=100]
    xn=len(df1.index)-len(df9.index)
    #
    xkwin,k0=-9,xtfb.staVars[0]
    if xn>k0:xkwin=0
    #
    return xkwin
           
def sta310_sta(xtfb,df):          
    xkwin=-9
    xk3,xk1,xk0=sta310_sta3(xtfb,df),sta310_sta1(xtfb,df),sta310_sta0(xtfb,df)
    if (xk3==3)and(xk1<0)and(xk0<1):xkwin=3
    if (xk3<1)and(xk1==1)and(xk0<0):xkwin=1
    if (xk3<1)and(xk1<0)and(xk0==0):xkwin=0   
    #
    #print('sta',xkwin,xk3,xk1,xk0)
    return xkwin
    
    

#------------- sta.ai.xxx


def sta_ai_log01(xtfb,df):
    #1
    xkwin,k00,k10,k30=-9,xtfb.staVars[0] ,xtfb.staVars[1],xtfb.staVars[2]
    ysgn='kwin'  #xtfb.ai_ysgn
    #2
    df[ysgn]=df[ysgn].astype(str)
    df[ysgn].replace('3','2', inplace=True)
    #3
    df[ysgn]=df[ysgn].astype(int)
    #4
    xtfb.ai_xdat,xtfb.ai_ydat= df[xtfb.ai_xlst],df[ysgn]  
    #5
    msgn=xtfb.ai_mx_sgn_lst[0] #'log'
    mx=zai.xmodel[msgn]
    dacc,df9=zai.mx_fun8mx(mx, xtfb.ai_xdat,xtfb.ai_ydat,yk0=1,fgInt=True) #,fgDebug=True
    #print('\n log01,dacc,',dacc)
    #6
    df3,df1,df0=df9[df9['y_pred']==2],df9[df9['y_pred']==1],df9[df9['y_pred']==0]
    dn3,dn1,dn0=len(df3.index),len(df1.index),len(df0.index)
    #7
    dn9,dsum=max(dn3,dn1,dn0),sum([dn3,dn1,dn0])
    #8
    if dsum>0:
        dk3,dk1,dk0=dn3/dsum*100,dn1/dsum*100,dn0/dsum*100
        if (dn3==dn9)and(dk3>k30):xkwin=3
        elif (dn1==dn9)and(dk1>k10):xkwin=1
        elif (dn0==dn9)and(dk0>k00):xkwin=0
        #
        #yk310=df9['y_test'][0]
        #xs0='@log01,{0}#,{1},xk,gid,{2},dsum.{3},dn310,{4},{5},{6},dk310,{7:.1f}%,{8:.1f}%,{9:.1f}%'
        #xss=xs0.format(xkwin,yk310,xtfb.kgid,dsum,dn3,dn1,dn0,dk3,dk1,dk0)
        #print(xss)
    
    #9
    return xkwin
    
    
