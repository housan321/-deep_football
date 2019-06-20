# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 

  
文件名:tfb_backtest.py
默认缩写：import tfb_backtest as tfbt
简介：Top极宽量化·回溯测试模块
 

'''
#
import copy
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
import zpd_talib as zta
#
import tfb_sys as tfsys
import tfb_tools as tft



def get_kelly_variance(odds_oz):
    ky_var = pd.DataFrame()
    cids = ['2','3', '4', '6', '293']
    odds_oz = odds_oz[odds_oz['cid'].isin(cids)]
    oz0 = ['plost0','pdraw0','pwin0']
    oz9 = ['plost9','pdraw9','pwin9']
    pb0 = ['vlost0','vdraw0','vwin0']
    pb9 = ['vlost9','vdraw9','vwin9']
                    
    odds_oz0 = odds_oz[oz0]
    odds_oz9 = odds_oz[oz9]
    prob0 = odds_oz[pb0]
    prob9 = odds_oz[pb9]
    odds_oz0 = odds_oz0.astype(float)
    odds_oz9 = odds_oz9.astype(float) 
    prob0 = prob0.astype(float)
    prob9 = prob9.astype(float)
    
    odds_oz0 = odds_oz0.values
    odds_oz9 = odds_oz9.values
    prob0 = prob0.values
    prob9 = prob9.values

    
    prob0_mean = np.mean(prob0, axis=0)
    prob9_mean = np.mean(prob9, axis=0)

    kelly0 = odds_oz0 * prob0_mean / 100
    kelly9 = odds_oz9 * prob9_mean / 100
    
    kelly0_mean = np.mean(kelly0, axis=0)
    kelly9_mean = np.mean(kelly9, axis=0)
    
    kelly0_var = (kelly0-kelly0_mean) * (kelly0-kelly0_mean) *100
    kelly9_var = (kelly9-kelly9_mean) * (kelly9-kelly9_mean) *100
    
    ky0_var = np.mean(kelly0_var, axis=0)
    ky9_var = np.mean(kelly9_var, axis=0)
       
#    ky0_var = pd.DataFrame(ky0_var)
    ky9_var = pd.DataFrame(ky9_var)

    ky_var = ky_var.append(ky9_var.T, ignore_index=True)
    
    ky_var = ky_var.round(4)

    
    return ky_var
    

def get_ratio_loss(odds_tz):
    '''
    volume_m = int((odds_tz['volume_m'])[0].replace(',',''))
    volume_d = int((odds_tz['volume_d'])[0].replace(',',''))
    volume_g = int((odds_tz['volume_g'])[0].replace(',',''))
    profit_m = int((odds_tz['profit_m'])[0].replace(',',''))
    profit_d = int((odds_tz['profit_d'])[0].replace(',',''))
    profit_g = int((odds_tz['profit_g'])[0].replace(',',''))
    '''
    odds_tz[odds_tz=='-'] = '0'
#    ratio = pd.DataFrame(columns=['ratio1','ratio2'])
    volume = odds_tz[['volume_g', 'volume_d', 'volume_m']]
    profit = odds_tz[['profit_g', 'profit_d', 'profit_m']]
    
    for index,row in volume.iterrows():
        row[0] = int(row[0].replace(',',''))
        row[1] = int(row[1].replace(',',''))
        row[2] = int(row[2].replace(',',''))
    for index,row in profit.iterrows():
        row[0] = int(row[0].replace(',',''))
        row[1] = int(row[1].replace(',',''))
        row[2] = int(row[2].replace(',',''))
    idx = np.argmin(profit.values, axis=1)
    loss = profit.min(axis=1)
    mask = copy.deepcopy(profit) 
    mask[mask>loss[0]] = 1
    mask[mask==loss[0]] = 0
    mask.columns = ['volume_g', 'volume_d', 'volume_m']
    
    vol_sum = volume.sum(axis=1)  
    ratio1 = abs(loss[0]) / vol_sum[0]
    
    vol_2 = (volume*mask).sum(axis=1)   
    ratio2 = abs(loss[0]) / vol_2[0]
    
    ratio = pd.DataFrame([[idx[0], abs(loss[0]), ratio1, ratio2]], columns=['idx', 'loss','loss/总','loss/分'])
    
#    ratio = ratio.append([ratio1, ratio2], ignore_index=True)

    ratio = ratio.round(4)
    
    return ratio



       
#----------fb.bt.misc.xxx
# 第一代CNN
def bt_lnkXDat1(g10,kcid):
    g20=pd.DataFrame(columns=tfsys.gidSgn)
    p_data=pd.DataFrame()
    kelly_var = pd.DataFrame()
    
    for i, row in g10.iterrows():
        gid=row['gid']
        fxdat_oz = tfsys.rxdat + gid + '_oz.dat'
        fxdat_az = tfsys.rxdat + gid + '_az.dat'
        if os.path.exists(fxdat_oz) and os.path.exists(fxdat_az):
            odds_oz = pd.read_csv(fxdat_oz, index_col = False, dtype = str, encoding = 'gb18030')  
            odds_az = pd.read_csv(fxdat_az, index_col = False, dtype = str, encoding = 'gb18030')
            
            ky_var = kelly_variance(odds_oz)
            
            if len(odds_oz) >= tfsys.cidrows and len(odds_az) >= tfsys.cidrows and odds_oz.loc[0, 'kwin'] == '-1' and odds_az.loc[0, 'kwin'] == '-1' :    #如果数据有CID_ROWS行并且有比赛结果才处理数据
                for sgn in tfsys.delSgn_oz:  
                    odds_oz.pop(sgn)           #删除一些没用的列  
                odds_oz.drop(odds_oz.index[tfsys.cidrows:], inplace=True)   #只保留前面CID_ROWS行
                    
                for sgn in tfsys.delSgn_az:  
                    odds_az.pop(sgn)           #删除一些没用的列  
                odds_az.drop(odds_az.index[tfsys.cidrows:], inplace=True)   #只保留前面CID_ROWS行
                    
                merge_data = pd.concat([odds_az, odds_oz], axis=1)                 
                p_data = p_data.append(merge_data, ignore_index=True)
                g20=g20.append(row.T,ignore_index=True)
                
                kelly_var = kelly_var.append(ky_var, ignore_index=True)

    return p_data, g20, kelly_var

# 第二代CNN，多输入模式CNN         
def bt_lnkXDat2(g10,kcid):
    g20=pd.DataFrame(columns=tfsys.gidSgn)  
    p_data = pd.DataFrame()
    kelly_var = pd.DataFrame()
    ratio_loss = pd.DataFrame()
    
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

    vb0 =['vback0']
    vb9 =['vback9']
    
    oz = ['pwin0','pwin9','pdraw0','pdraw9','plost0','plost9']
    
    kali_columns = ['vwinkali','vdrawkali','vlostkali']
    oz_columns = ['pwin','pdraw','plost']
    az_columns = ['mshui', 'gshui', 'pan'] 
      
    for i, row in g10.iterrows():
        gid = row['gid']
        fxdat_oz = tfsys.rxdat + gid + '_oz.dat'
        fxdat_az = tfsys.rxdat + gid + '_az.dat'
        fxdat_tz = tfsys.rxdat + gid + '_tz.dat'
        if os.path.exists(fxdat_oz) and os.path.exists(fxdat_az):
            odds_oz = pd.read_csv(fxdat_oz, index_col = False, dtype = str, encoding = 'gb18030')  
            odds_az = pd.read_csv(fxdat_az, index_col = False, dtype = str, encoding = 'gb18030')      
            odds_tz = pd.read_csv(fxdat_tz, index_col = False, dtype = str, encoding = 'gb18030')  
            
            if len(odds_oz) >= tfsys.cidrows and len(odds_az) >= tfsys.cidrows and len(odds_tz) == 1 and odds_oz.loc[0, 'kwin'] == '-1' and odds_az.loc[0, 'kwin'] == '-1' :    #如果数据有CID_ROWS行并且有比赛结果才处理数据
                ky_var = get_kelly_variance(odds_oz)
                ratio = get_ratio_loss(odds_tz)
                
                odds_az = odds_az[0:tfsys.cidrows]
                odds_oz = odds_oz[1:tfsys.cidrows+1]

                odds_oz = odds_oz[oz_cols].astype(float)
                odds_az = odds_az[az_cols].astype(float) 
                #1
                odds_oz0_9 = odds_oz[oz]
                odds_oz0 = odds_oz[oz0]
                odds_oz9 = odds_oz[oz9]
                odds_az0 = odds_az[az0]
                odds_az9 = odds_az[az9]
                '''
                odds_oz0.columns = oz_columns
                odds_oz9.columns = oz_columns
                odds_az0.columns = az_columns
                odds_az9.columns = az_columns                
                az_diff = odds_az9 - odds_az0
                oz_diff = odds_oz9 - odds_oz0
                '''
                #2
                odds_kali0 = odds_oz[kali0]
                odds_kali9 = odds_oz[kali9]
                odds_kali0.columns = kali_columns
                odds_kali9.columns = kali_columns
                kali_diff = odds_kali9 - odds_kali0 
                
                #3
                vback0 = odds_oz['vback0'] /100
                vback9 = odds_oz['vback9'] /100                   
                vback0_diff = odds_kali0.sub(vback0,axis=0)
                vback9_diff = odds_kali9.sub(vback9,axis=0)                
                vb_diff = vback9 - vback0
                '''    
                az_diff = az_diff.reset_index(drop=True)
                oz_diff = oz_diff.reset_index(drop=True)
                '''
                kali_diff = kali_diff.reset_index(drop=True)
                vback0_diff = vback0_diff.reset_index(drop=True)
                vback9_diff = vback9_diff.reset_index(drop=True)

                odds_az0 = odds_az0.reset_index(drop=True)
                odds_oz0 = odds_oz0.reset_index(drop=True) 
                odds_az9 = odds_az9.reset_index(drop=True)
                odds_oz9 = odds_oz9.reset_index(drop=True)
                odds_oz0_9 = odds_oz0_9.reset_index(drop=True) 
                vb_diff = vb_diff.reset_index(drop=True) 
                
                #
                merge_data = pd.concat([odds_az0, odds_oz0, odds_az9, odds_oz9, kali_diff, vb_diff, vback0_diff, vback9_diff], axis=1)                 
                p_data = p_data.append(merge_data, ignore_index=True)
                
                g20=g20.append(row.T,ignore_index=True)
                
                kelly_var = kelly_var.append(ky_var, ignore_index=True)
                ratio_loss = ratio_loss.append(ratio, ignore_index=True)
            
    return p_data, g20, kelly_var, ratio_loss


#################################################################################
col_sgn = ['win0','draw0','lost0','back0','win9','draw9','lost9', 'back9', 'kelly3', 'kelly1', 'kelly0']
n_step = 64 #赔率变化的次数
    
def align_odds(odds_list):
    odds_list = odds_list.sort_index(ascending=False)
    len_ls = len(odds_list)
    if len_ls > n_step:
        odds_list = odds_list[0:n_step]
    elif len_ls < n_step:
         last_step = odds_list[-1:]
         for i in range(n_step-len_ls):
             odds_list = odds_list.append(last_step, ignore_index=True)
    
    return odds_list


def bt_lnkXDat_LSTM(g10):
    sgn1 = ['gset','mplay','gplay']
    sub_samples = pd.DataFrame()
    match_name = pd.DataFrame()
    
    samples = tfsys.samples
    if len(samples) == 0: return sub_samples, match_name  ##没有合适的比赛，返回
    
    sub_gids = pd.merge(g10[['gid']], samples[['gid']], on=['gid'])
    if len(sub_gids) == 0: return sub_samples, match_name   ##没有合适的比赛，返回
    
    for i in range(len(sub_gids['gid'])): 
        sub_samples = sub_samples.append(samples[samples['gid'] == sub_gids['gid'][i]], ignore_index=True)
        match_name = match_name.append(g10[g10['gid'] == sub_gids['gid'][i]], ignore_index=True)
    
    sub_samples.pop('gid')
    match_name = match_name[sgn1]
   
    return sub_samples, match_name


#####################################################################################





#----------fb.bt.1day.xxx
    
def bt_1d_ret(xtfb):
    #print('\n@bt_1d_ret')
    ret1d=pd.Series(tfsys.retNil,index=tfsys.retSgn) 
    ret1d['xtim'],ret1d['cid'],ret1d['num9']=xtfb.ktimStr,xtfb.kcid,0
    #     
    df9=xtfb.poolDay;
    xnum=len(df9.index)
    if xnum==0:return ret1d
    #     
    nlst,flst=['kwin','kwin_sta'],['pwin9','pdraw9','plost9']
    tft.fb_df_type4mlst(df9,nlst,flst)
    #
    for i, row in df9.iterrows():
        kwin2=row['kwin_sta']

        rsgn='num'+str(kwin2)
        ret1d['num9'],ret1d[rsgn]=ret1d['num9']+1,ret1d[rsgn]+1
        #

        dmoney=tft.fb_kwin2pdat(kwin2,row)
        rsgn='nwin'+str(kwin2)     
        ret1d['nwin9'],ret1d[rsgn]=ret1d['nwin9']+1,ret1d[rsgn]+1
        rsgn='ret'+str(kwin2)          
        ret1d['ret9'],ret1d[rsgn]=ret1d['ret9']+dmoney,ret1d[rsgn]+dmoney
                #print(i,'#1',kwin,dmoney)
    #
    xlst=[9,3,1,0]
    for xd in xlst:
        xss=str(xd)
        dn=ret1d['num'+xss]
        if dn>0:
            ret1d['kret'+xss]=round(ret1d['ret'+xss]/dn*100,2)
            ret1d['knum'+xss]=round(ret1d['nwin'+xss]/dn*100,2)
    
    #
    #print(ret1d);
    return ret1d    
    
def bt_1d_anz_1play(xtfb):
    #print('\nbt_1d_anz_1play')
    bars=xtfb.bars
    gid=bars['gid']
    xtfb.kgid=gid
    df=xtfb.xdat10[xtfb.xdat10.gid==gid]
    xkwin, ret_pr=xtfb.funSta(xtfb, df)
    #---trade
    if xkwin!=-9:
        xtfb.poolInx.append(gid)
        #
        g10=bars
        c10=df[df.cid==xtfb.kcid].copy()
        c10=c10.reset_index(drop=True)
        #
        g10['kwin_sta'],g10['cid']=xkwin,xtfb.kcid
        g10['pwin9'],g10['pdraw9'],g10['plost9']=c10['pwin9'][0],c10['pdraw9'][0],c10['plost9'][0]
        #
        xtfb.poolDay=xtfb.poolDay.append(g10.T,ignore_index=True)
        xtfb.poolTrd=xtfb.poolTrd.append(g10.T,ignore_index=True)
        #print(xtfb.poolDay)
#        xtfb.probability=xtfb.probability.append(ret_pr, ignore_index=True)
#        print(ret_pr)
        
def bt_1d_anz(xtfb):
    # 1#day
    #
    for i, row in xtfb.gid10.iterrows():
        xtfb.bars=row
        #
        bt_1d_anz_1play(xtfb)
    #            
    if len(xtfb.poolDay.index)>0:
        ret01=bt_1d_ret(xtfb)
        if ret01['num9']>0:
            xtfb.poolRet=xtfb.poolRet.append(ret01.T,ignore_index=True)
            #print(xtfb.poolRet)
        
def bt_1dayMain(xtfb):        
    xtfb.poolInx,xtfb.xdat10=[],None
    xtfb.poolDay=pd.DataFrame(columns=tfsys.poolSgn)
    #  
    df=tfsys.gids    
    #

    g10=df[df.tplay==xtfb.ktimStr]
              
    #------  lnk.xdat
#    xdatRR,gid10RR=bt_lnkXDat(g10,xtfb.kcid)   # xtfb.gid10：赛事ID号
#    xdat,xtfb.gid10, kelly_var = bt_lnkXDat1(g10,xtfb.kcid)   # xtfb.gid10：赛事ID号
#    xdat,xtfb.gid10, kelly_var, ratio_loss = bt_lnkXDat2(g10,xtfb.kcid)   # xtfb.gid10：赛事ID号
    xdat, match_name = bt_lnkXDat_LSTM(g10)
    if (len(xdat)==0) or (len(match_name)==0): return 
   
    if len(xdat.index)>0:
        #--dat.pre0
#        xlst=['pwin0','pdraw0','plost0','pwin9','pdraw9','plost9']
#        tft.fb_df_type2float(xdat,xlst)
#        tft.fb_df_type_xed(xdat)
        #
        xtfb.xdat10=xdat          # xtfb.xdat10：赛事赔率数据
        #
        #2  data.pre
#        xtfb.funPre(xtfb)     #原程序是预处理部分，（***但我这里用来作预测了**）        
        xtfb.funPre(xtfb, match_name)     #原程序是预处理部分，（***但我这里用来作预测了**）
        #fss='tmp/df901b.dat'
        #xtfb.xdat10.to_csv(fss,index=False,encoding='gb18030')  
        #
        #3  bt.anz&trade #gid -->pool
#        bt_1d_anz(xtfb)   #原程序是预测部分，处理我不需要了，留作以后开发
        #   
    
#----------bt--main

    
def bt_main(xtfb,timStr):
    if timStr=='':ktim=xtfb.tim_now
    else:ktim=arrow.get(timStr)
    #
    nday=tfsys.xnday_down
    #
    tfsys.gids['kwin_sta']=-9
    xtfb.poolRet=pd.DataFrame(columns=tfsys.retSgn)
    for tc in range(nday):  #预测天数
        xtim=ktim.shift(days = tc)
        xtimStr=xtim.format('YYYY-MM-DD')
        print('\n',tc,'#',xtimStr)
        #
        if xtim<xtfb.tim0_gid:break
        #
        xtfb.ktimStr=xtimStr
        bt_1dayMain(xtfb)
        #

def bt_main_ret(xtfb,fgMsg=False):
    #1
    ret9=pd.Series(tfsys.retNil,index=tfsys.retSgn) 
    rlst=tfsys.retSgn[1:]
    for rsgn in rlst:
        ret9[rsgn]=xtfb.poolRet[rsgn].sum()
        #print(rsgn,r10[rsgn].sum())
    #2
    xlst=[9,3,1,0]
    for xd in xlst:
        xss=str(xd)
        dn=ret9['num'+xss]
        if dn>0:
            #！！！ kret=sum(ret)/num9，not avg（ret）
            ret9['kret'+xss]=round(ret9['ret'+xss]/dn*100,2)  
            ret9['knum'+xss]=round(ret9['nwin'+xss]/dn*100,2)
    #3  
    nlst=['num9','nwin9','num3','nwin3','num1','nwin1','num0','nwin0']
    float_lst=['kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0', 'ret9','ret3','ret1','ret0']
    tft.fb_df_type4mlst(xtfb.poolRet,nlst,float_lst)
    for xsgn in float_lst:
        xtfb.poolRet[xsgn]=round(xtfb.poolRet[xsgn],2)
        ret9[xsgn]=round(ret9[xsgn],2)
    #4
    #--save.dat
    ret9['xtim'],ret9['cid']='sum',xtfb.kcid
    xtfb.poolRet=xtfb.poolRet.append(ret9,ignore_index=True)
    xtfb.poolTrd.to_csv(xtfb.poolTrdFN,index=False,encoding='gb18030')
    xtfb.poolRet.to_csv(xtfb.poolRetFN,index=False,encoding='gb18030')
    
    #5
    if fgMsg:
        print('\nxtfb.poolTrd,足彩推荐')
        print(xtfb.poolTrd.head())
        print('\nxtfb.poolRet，回报率汇总')
        print(xtfb.poolRet.tail())











    