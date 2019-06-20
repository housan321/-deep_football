# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 
 
文件名:tfb_tools.py
默认缩写：import tfb_tools as tft
简介：Top极宽量化·常用足彩工具函数集

''' 


import os,sys,io,re
import random,arrow,bs4
import numpy as np
import numexpr as ne
import pandas as pd
import tushare as ts
import requests
from bs4 import BeautifulSoup
import time
import datetime
import re

#
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor,as_completed
#from concurrent.futures import ProcessPoolExecutor
#
#import inspect
#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
#
import tfb_sys as tfsys
import tfb_strategy as tfsty

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#
#-----------------------
'''
var&const
tfb.init.obj
tfb.misc
#
tfb.get.dat.xxx
#
tfb.dat.xxx

'''

#-----------------------
#----------var&const

def fb_df_type_xed(df):
    df['qj']=df['qj'].astype(int)
    df['qs']=df['qs'].astype(int)
    df['qr']=df['qr'].astype(int)
    df['kwin']=df['kwin'].astype(int)
    df['kwinrq']=df['kwinrq'].astype(int)

def fb_df_type2float(df,xlst):
    for xsgn in xlst:
        df[xsgn]=df[xsgn].astype(float)

def fb_df_type4mlst(df,nlst,flst):
    for xsgn in nlst:
        df[xsgn]=df[xsgn].astype(int)
        
    for xsgn in flst:
        df[xsgn]=df[xsgn].astype(float)
    
#----------tfb.init.obj
     
def fb_init(rs0='/tfbDat/',fgid=''):
    #1
    xtfb=tfsys.zTopFoolball()
    xtfb.tim_now=arrow.now()
    xtfb.timStr_now=xtfb.tim_now.format('YYYY-MM-DD')
    xtfb.tim0,xtfb.tim0Str=xtfb.tim_now,xtfb.timStr_now
    print('now:',zt.tim_now_str())
    
    #2
    #xtfb.pools=[]
    xtfb.kcid='1'  #官方,3=Bet365
    xtfb.funPre=tfsty.sta00_pre
    xtfb.funSta=tfsty.sta00_sta
    #
    xss=xtfb.timStr_now
    xtfb.poolTrdFN,xtfb.poolRetFN='log\poolTrd_'+xss+'.csv','log\poolRet_'+xss+'.csv'
    #3
    if rs0!='':
        tfsys.rdat=rs0
        tfsys.rxdat=rs0+'xdat/'
        tfsys.rhtmOuzhi=rs0+'xhtm/js_oz/'
        tfsys.rhtmYazhi=rs0+'xhtm/htm_az/'
        tfsys.rhtmShuju=rs0+'xhtm/htm_sj/'
        
    #4
    if fgid!='':
        tfsys.gidsFN=fgid
        #xtfb.gids=pd.read_csv(fgid,index_col=0,dtype=str,encoding='gbk')
        tfsys.gids=pd.read_csv(fgid,index_col=False,dtype=str,encoding='gb18030')
        fb_df_type_xed(tfsys.gids)
        tfsys.gidsNum=len(tfsys.gids.index)
        #-----tim.xxx
        xtfb.gid_tim0str,xtfb.gid_tim9str=tfsys.gids['tplay'].min(),tfsys.gids['tplay'].max()
        tim0,tim9=arrow.get(xtfb.gid_tim0str),arrow.get(xtfb.gid_tim9str)
        xtfb.gid_nday,xtfb.gid_nday_tim9=zt.timNDay('',tim0),zt.timNDay('',tim9)
        print('gid tim0: {0}, nday: {1}'.format(xtfb.gid_tim0str,xtfb.gid_nday))    
        print('gid tim9: {0}, nday: {1}'.format(xtfb.gid_tim9str,xtfb.gid_nday_tim9))    
       
        
    #
    return xtfb 
       
#----------tfb.misc
def fb_tweekXed(tstr):
    str_week=['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
    str_inx=['1','2','3','4','5','6','0']
    tstr=zstr.str_mxrep(tstr,str_week,str_inx)
    #
    return tstr
            
def fb_kwin4qnum(jq,sq,rq=0):
    if (jq<0)or(sq<0):return -1
    #   
    jqk=jq+rq  #or -rq
    if jqk>sq:kwin=3     #主要用于分类，表示第2类
    elif jqk<sq:kwin=0
    else:kwin=1
    #
    return kwin

def fb_kwin2pdat(kwin,ds):
    if kwin==3:xd=ds['pwin9']
    elif kwin==1:xd=ds['pdraw9']
    elif kwin==0:xd=ds['plost9']
    #
    return xd    
    
#----------tfb.get.dat.xxx
#def fb_tweekXed(tstr):
             
def fb_gid_get4htm(htm):
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    df=pd.DataFrame(columns=tfsys.gidSgn,dtype=str)
    ds=pd.Series(tfsys.gidNil,index=tfsys.gidSgn,dtype=str)
    
    #---1#
#    zsys.bs_get_ktag_kstr=['align','right']
    zsys.bs_get_ktag_kstr='matchid'
    x10=bs.find_all(zweb.bs_get_ktag)
    for xc,x in enumerate(x10):
        ds=pd.Series(tfsys.gidNil,index=tfsys.gidSgn,dtype=str)
        #print('\n@x\n',xc,'#',x.attrs)
        
        home_team = x.find(attrs={'align':'right'})
        guest_team = x.find(attrs={'align':'left'})
        ds['gid'] = home_team.find('span')['id'].split('_')[1]
        ds['gset'] = x['gamename']
        ds['mplay'] = home_team.text
        ds['gplay'] = guest_team.text
        
        clst = zt.lst4objs_txt2(zstr.str_fltHtm(x.text), ['\n','\t','%'])
        score = clst[5].split('-')
        if score[0]:
            ds['qj'] = score[0]
        if score[1]:
            ds['qs'] = score[1]    
                
        date = x.find(attrs={'title':re.compile("截止时间:*")})
        if date.text == '完场':
            ds['kend'] = '1'
        ds['tsell'] = date['title'].split('：')[1]
        ds['tplay'] = ds['tsell'].split(' ')[0]
        ds['tweek'] = x['name']
          
        kwin=fb_kwin4qnum(int(ds['qj']),int(ds['qs']))
        ds['kwin']=str(kwin) 

        #
        df=df.append(ds.T,ignore_index=True)
  
    #---5#
    df=df[df['gid']!='-1']
    return df

###############################################################################
# 一个赛季所有场次的比赛的gid
def fb_league_gids(htm, league, fgExt=True):
    df=pd.DataFrame(columns=tfsys.gidSgn,dtype=str)
    ds=pd.Series(tfsys.gidNil,index=tfsys.gidSgn,dtype=str)
 
    nround = re.findall(r"jh\[\"R_.*\"\] = \[.*\]", htm)  #赛季有多少轮   
    n_round = len(nround)
    
    for n in range(5, n_round): ##从第5轮开始采集数据
        pattern = "jh\[\"R_" + str(n+1) + "\"\] = \[.*\]"    
        result = re.findall(pattern, htm) 
        res = result[0].split(' = ')
        games = re.findall(r"\[\[(.*)\]\]", res[1])
        games = games[0].split('],[')
    
        for game in games:
            res = re.split("\,", game)
            ds['gid'] = res[0]
            ds['gset'] = league
            ds['kend'] = '1' #比赛已结束
            score = res[6]
            score = zstr.str_flt(score, '\'')
            ds['qj'] = re.split("\-", score)[0]
            ds['qs'] = re.split("\-", score)[1]
            qj = int(ds['qj'])
            qs = int(ds['qs'])
            if qj > qs:
                ds['kwin'] = '3'
            elif qj < qs:
                ds['kwin'] = '0'
            else:
                ds['kwin'] = '1'
            ds['mplay'] = tfsys.teamIds[res[4]]
            ds['gplay'] = tfsys.teamIds[res[5]]
            ds['mtid'] = res[4]
            ds['gtid'] = res[5]
            ds['tplay'] = res[3]
            ds['tweek'] = str(n+1)  #第几轮赛事 
            ds['tsell'] = str(n+1)
            df=df.append(ds.T,ignore_index=True)


    if fgExt:fb_gid_getExt(df)    #单线程
    else: fb_gid_getExtPool(df)   #多线程

    if tfsys.gidsFN!='':
        print('+++++')
        print(tfsys.gids.tail())
        tfsys.gids.to_csv(tfsys.gidsFN,index=False,encoding='gb18030')

    

###############################################################################
def get_sign(result):
    if result == '0':
        return 'W'
    elif result == '1':
        return 'D'
    elif result == '2':
        return 'L'
    else:
        return 'M'
#####################################################
        
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
######################################################### 
        
# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum    
###############################################################
    
# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[0:3] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[0:3] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0
 
### one hot 编码    
def get_encode(result):
    if result[0] == 'D':
        return 0
    elif result[0] == 'L':
        return 1    
    elif result[0] == 'M':
        return 2 
    elif result[0] == 'W':
        return 3
    else:
        return 2
  

def fb_gid_getExt_az4clst(ds,clst):
    i=0;
    ds['mshui0'],ds['pan0'],ds['gshui0']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['mshui9'],ds['pan9'],ds['gshui9']=clst[i],clst[i+1],clst[i+2]
    #
    return ds    
    
 
def fb_gid_getExt_oz4clst(ds,clst):
    i=0;
    ds['pwin0'],ds['pdraw0'],ds['plost0']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['pwin9'],ds['pdraw9'],ds['plost9']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin0'],ds['vdraw0'],ds['vlost0']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin9'],ds['vdraw9'],ds['vlost9']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vback0'],ds['vback9']=clst[i],clst[i+1]
    i=i+2;
    ds['vwin0kali'],ds['vdraw0kali'],ds['vlost0kali']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin9kali'],ds['vdraw9kali'],ds['vlost9kali']=clst[i],clst[i+1],clst[i+2]
    #
    return ds
  
def fb_gid_getExt_tz4clst(ds,clst):
    ds['av_odd_h'],ds['av_odd_d'],ds['av_odd_g']=clst[17],clst[30],clst[41]
    ds['av_prob_h'],ds['av_prob_d'],ds['av_prob_g']=clst[18],clst[31],clst[42]     
    ds['av_prob_h'] = ds['av_prob_h'].replace('%', '')
    ds['av_prob_d'] = ds['av_prob_d'].replace('%', '')
    ds['av_prob_g'] = ds['av_prob_g'].replace('%', '')
    
    ds['av_back']=clst[19]  
    ds['av_back']=ds['av_back'].replace('%', '')
    
    ds['bf_odd_h'],ds['bf_odd_d'],ds['bf_odd_g']=clst[20],clst[32],clst[43]  
    ds['bf_prob_h'],ds['bf_prob_d'],ds['bf_prob_g']=clst[21],clst[33],clst[44] 
    
    ds['bf_prob_h'] = ds['bf_prob_h'].replace('%', '')
    ds['bf_prob_d'] = ds['bf_prob_d'].replace('%', '')
    ds['bf_prob_g'] = ds['bf_prob_g'].replace('%', '')
    
    ds['bf_back']=clst[22]
    ds['bf_back']=ds['bf_back'].replace('%', '')
    
    ds['volume_h'],ds['volume_d'],ds['volume_g']=clst[23],clst[34],clst[45] 
    ds['volume_h'] = ds['volume_h'].replace(',', '')
    ds['volume_d'] = ds['volume_d'].replace(',', '')
    ds['volume_g'] = ds['volume_g'].replace(',', '')
    
    ds['vol_ratio_h'],ds['vol_ratio_d'],ds['vol_ratio_g']=clst[24],clst[35],clst[46]  
    ds['vol_ratio_h'] = ds['vol_ratio_h'].replace('%', '')
    ds['vol_ratio_d'] = ds['vol_ratio_d'].replace('%', '')
    ds['vol_ratio_g'] = ds['vol_ratio_g'].replace('%', '')
    
#    print(clst[36])
    ds['shui_h'],ds['pan'],ds['shui_g']=clst[25], tfsys.az_pan[clst[36]], clst[47] 

    ds['profit_h'],ds['profit_d'],ds['profit_g']=clst[26],clst[37],clst[48]   
    ds['pf_idx_h'],ds['pf_idx_d'],ds['pf_idx_g']=clst[27],clst[38],clst[49]
    ds['hot_idx_h'],ds['hot_idx_d'],ds['hot_idx_g']=clst[28],clst[39],clst[50]
    
    #
    return ds    


###  提取多家主流菠菜公司初盘、晚盘赔率
def fb_gid_getExt_oz4htm_1(htm,bars,ftg=''):
    col_sgn = ['win0','draw0','lost0','back0','win9','draw9','lost9', 'back9', 'kelly_w', 'kelly_d', 'kelly_l']
    odds_list = pd.DataFrame(columns=col_sgn)
    
    result = re.findall(r"var game=Array(.*)", htm)
    result = re.findall(r"\(\"(.*)\"\)", result[0])
    if len(result) < 1:
        print("---------------  ---------------------")
        return
    result = re.split("\",\"", result[0])  
    for cid in tfsys.cids:      
        for value in result:
            res = re.split("\|", value)
            if res[0] == cid:  #找到该菠菜公司
                data = res[3:6] + res[9:13] + res[16:20]
                if res[10] == '' or res[11] == '' or res[16] == '':
                    data[4:8] = data[0:4]    
                data = pd.Series(data, index=col_sgn)
                odds_list = odds_list.append(data, ignore_index=True)
                break
#    odds_list = align_odds(odds_list)    
    odds_list = odds_list.iloc[0:32,:] ##只取32家菠菜公司赔率
    if ftg!='':odds_list.to_csv(ftg,index=False,encoding='gb18030')   
  
    return odds_list
    
 
### 提取某家菠菜公司赔率变化序列    
def fb_gid_getExt_oz4htm_2(htm,bars,ftg=''):    
    col_sgn = ['win','draw','lost','time','winkelly','drawkelly','lostkelly']
    cname = 'William Hill'
    odds_list = pd.DataFrame(columns=col_sgn)
    cid = -1
    odds = -1
 
    result = re.findall(r"var game=Array(.*)", htm)
    result = re.findall(r"\(\"(.*)\"\)", result[0])
    if len(result) < 1:
        print("---------------  ---------------------")
        return
    result = re.split("\",\"", result[0])  
    for value in result:
        res = re.split("\|", value)
        if res[2] == cname:   
            cid = res[1]
            break
    # 读取赔率序列    
    if cid == -1: return    
    datas = re.findall(r"var gameDetail=Array(.*)", htm)
    datas = re.findall(r"\(\"(.*)\"\)", datas[0])
    if len(datas) < 1: 
        print("@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@")
        return
    datas = re.split("\",\"", datas[0])
    for value in datas:
        res = re.split("\^", value)
        if res[0] == cid:
            odds = re.split(";", res[1][:-1])
            break
    
    if odds != -1:
        for value in odds:
            col = re.split("\|", value)
            if len(col) != len(col_sgn): continue
            step = pd.Series(re.split("\|", value), index=col_sgn)
            odds_list = odds_list.append(step, ignore_index=True)
    else: return
   
    if ftg!='':odds_list.to_csv(ftg,index=False,encoding='gb18030')
    
    return odds_list



##########################################################################
########################################################################## 
def lst4objs_txt_az(xobjs,fltLst=[]):
    clst=[]
    for x in xobjs:
        #css=x.text.replace('\n','')
        css=zstr.str_flt(x.get_text(),fltLst)
        c20=css.split(' ')    
        for c in c20:
            if c!='' and c!='升' and c!='降':
                clst.append(c)
    cl = clst[0:3]+clst[-3:]
    cl[1] = tfsys.pan[cl[1]]
    cl[4] = tfsys.pan[cl[4]]
    return cl  


def fb_gid_getExt_az4htm(htm,bars,ftg=''):  
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    x10=bs.find_all('tr',xls='row')
    df=pd.DataFrame(columns=tfsys.gxdatSgn_az)
    ds=pd.Series(tfsys.gxdatNil_az,index=tfsys.gxdatSgn_az)
    xc,gid=0,bars['gid']

    xlst=['gset','mplay','mtid','gplay','gtid', 'qj','qs','qr','kwin','kwinrq','tplay','tweek']
    for xc,x in enumerate(x10):
        #print('\n@x\n',xc,'#',x.attrs)
        x2=x.find('td',class_='tb_plgs');#print(x2.attrs)
        ds['gid'],ds['cid']=gid,x['id']
        cname = x2.get_text()
        cname = cname[0: int(len(cname)/2)]
        ds['cname'] = cname
        #
        x20=x.find_all('table',class_='pl_table_data');
        clst=lst4objs_txt_az(x20,['\n','\t','%', '↓', '↑'])
        ds=fb_gid_getExt_az4clst(ds,clst)
        #
        zdat.df_2ds8xlst(bars,ds,xlst)
        df=df.append(ds.T,ignore_index=True)

    if ftg!='':df.to_csv(ftg,index=False,encoding='gb18030')

    return df    

  
def fb_gid_getExt_tz4htm(htm,bars,ftg=''):
    bs = BeautifulSoup(htm, 'html.parser')
    tableList = bs.select("table")
    df = pd.DataFrame(columns=tfsys.gxdatSgn_tz)
    ds = pd.Series(tfsys.gxdatNil_tz,index=tfsys.gxdatSgn_tz)
    qj = bars['qj']
    qs = bars['qs']
    kwin = bars['kwin'] 
    
    if len(tableList) != 6: ####网页数据有误
        return df
    
    tdArr = []
    for td in tableList[4].select("td"):
        tdArr.append(td.get_text())
                
    ds=fb_gid_getExt_tz4clst(ds, tdArr) 
    df=df.append(ds.T,ignore_index=True) 
    df['qj'] = qj
    df['qs'] = qs
    df['kwin'] = kwin
    if df.isnull().values.any(): return ##如果有Nan值就不保存该数据
    if ftg!='':df.to_csv(ftg,index=False,encoding='gb18030')
    #
    return df
    

################################################################################## 
##################################################################################
    


    
        
    


#-----tfb.dat.xxx
def fb_xdat_xrd020(fsr,xlst,ysgn='kwin',k0=1,fgPr=False):    
    
    #1
    df=pd.read_csv(fsr,index_col=False,encoding='gb18030')
    #2
    if ysgn=='kwin':
        df[ysgn]=df[ysgn].astype(str)
        df[ysgn].replace('3','2', inplace=True)
        #df['kwin'].replace('3','2', inplace=True)
    #3
    df[ysgn]=df[ysgn].astype(float)
    df[ysgn]=round(df[ysgn]*k0).astype(int)              
    #4              
    x_dat,y_dat= df[xlst],df[ysgn]   
      
    #5
    if fgPr:
        print('\n',fsr);
        print('\nx_dat');print(x_dat.tail())
        print('\ny_dat');print(y_dat.tail())
        #df.to_csv('tmp\df.csv',index=False,encoding='gb18030')
    #6
    return  x_dat,y_dat     
        
def fb_xdat_xlnk(rs0,ftg):
    flst=zt.lst4dir(rs0)
    df9=pd.DataFrame(columns=tfsys.gxdatSgn,dtype=str)
    for xc,fs0 in enumerate(flst):
        fss=rs0+fs0
        print(xc,fss)
        df=pd.read_csv(fss,index_col=False,dtype=str,encoding='gb18030')
        #
        df2=df[df['kwin']!='-1']
        df9=df9.append(df2,ignore_index=True)
        #
        if (xc % 2000)==0:
            #df9.to_csv(ftg,index=False,encoding='gb18030')
            fs2='tmp/x_'+str(xc)+'.dat';print(fs2,fss)
            df9.to_csv(fs2,index=False,encoding='gb18030')
    #
    df9.to_csv(ftg,index=False,encoding='gb18030')  















def HM_one_hot_encoder(data):
    data.HM1=get_encode(data.HM1)
    data.HM2=get_encode(data.HM2)
    data.HM3=get_encode(data.HM3)
    data.HM4=get_encode(data.HM4)
    data.HM5=get_encode(data.HM5)
   
    data.AM1=get_encode(data.AM1)
    data.AM2=get_encode(data.AM2)
    data.AM3=get_encode(data.AM3)
    data.AM4=get_encode(data.AM4)
    data.AM5=get_encode(data.AM5)
    
    data=data.drop(["HM5","AM5"],axis=1)
    
    return data



def fb_get_features(htm,bars,ftg=''):
    cols = ['HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 
            'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 
            'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
            'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
            'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
            'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'teamPL']
    features = pd.DataFrame(columns=cols)
    fea = pd.Series(index=cols,dtype=str)
    
    game = re.findall(r"var ScheduleID=.*", htm)
    if len(game)==0:
        game = re.findall(r"ScheduleID=.*", htm)
    game = zstr.str_flt(game[0], [';',' ','\n','\t','\r'])
    gid = (re.split("\=", game))[1]
    
    home = re.findall(r"var hometeamID=.*", htm)
    home = zstr.str_flt(home[0], [';',' ','\n','\t','\r'])
    homeId = (re.split("\=", home))[1]

    away = re.findall(r"var guestteamID=.*", htm)
    away = (re.split("\=", away[0]))[1]
    awayId = zstr.str_flt(away, [';',' ','\n','\t','\r'])

    league_sc = tfsys.league_sc
    hIndex = np.where(league_sc['team_id'] == homeId)
    aIndex = np.where(league_sc['team_id'] == awayId)
    if len(hIndex[0])==0 and len(aIndex[0])==0: 
        return
    
    hIndex = league_sc.index[hIndex][0] 
    aIndex = league_sc.index[aIndex][0]

     
    fea['HTGS'] = league_sc.iloc[hIndex].goal_score
    fea['ATGS'] = league_sc.iloc[aIndex].goal_score
    fea['HTGC'] = league_sc.iloc[hIndex].goal_conceded
    fea['ATGC'] = league_sc.iloc[aIndex].goal_conceded
    fea['HTP'] = league_sc.iloc[hIndex].TP / league_sc.iloc[hIndex].MW
    fea['ATP'] = league_sc.iloc[aIndex].TP / league_sc.iloc[hIndex].MW
    
    fea['HM1'] = league_sc.iloc[hIndex].M1
    fea['HM2'] = league_sc.iloc[hIndex].M2
    fea['HM3'] = league_sc.iloc[hIndex].M3
    fea['HM4'] = league_sc.iloc[hIndex].M4
    fea['HM5'] = league_sc.iloc[hIndex].M5
    fea['AM1'] = league_sc.iloc[aIndex].M1
    fea['AM2'] = league_sc.iloc[aIndex].M2
    fea['AM3'] = league_sc.iloc[aIndex].M3
    fea['AM4'] = league_sc.iloc[aIndex].M4
    fea['AM5'] = league_sc.iloc[aIndex].M5
    
    HTFormPtsStr = league_sc.iloc[hIndex].M1 + league_sc.iloc[hIndex].M2 + league_sc.iloc[hIndex].M3 + league_sc.iloc[hIndex].M4 + league_sc.iloc[hIndex].M5
    ATFormPtsStr = league_sc.iloc[aIndex].M1 + league_sc.iloc[aIndex].M2 + league_sc.iloc[aIndex].M3 + league_sc.iloc[aIndex].M4 + league_sc.iloc[aIndex].M5  
    
    fea['HTWinStreak3'] = get_3game_ws(HTFormPtsStr)
    fea['HTWinStreak5'] = get_5game_ws(HTFormPtsStr)
    fea['HTLossStreak3'] = get_3game_ls(HTFormPtsStr)
    fea['HTLossStreak5'] = get_5game_ls(HTFormPtsStr)   
    fea['ATWinStreak3'] = get_3game_ws(ATFormPtsStr)
    fea['ATWinStreak5'] = get_5game_ws(ATFormPtsStr)
    fea['ATLossStreak3'] = get_3game_ls(ATFormPtsStr)
    fea['ATLossStreak5'] = get_5game_ls(ATFormPtsStr)
     
    fea['HTGD'] = league_sc.iloc[hIndex].goal_diff / league_sc.iloc[hIndex].MW
    fea['ATGD'] = league_sc.iloc[aIndex].goal_diff / league_sc.iloc[aIndex].MW
    fea['DiffPts'] = (league_sc.iloc[hIndex].TP - league_sc.iloc[aIndex].TP) / league_sc.iloc[aIndex].MW   

    HTFormPts = get_form_points(HTFormPtsStr)
    ATFormPts = get_form_points(ATFormPtsStr)  
    DiffFormPts = HTFormPts - ATFormPts
    fea['DiffFormPts'] = DiffFormPts / league_sc.iloc[aIndex].MW
    fea['teamPL'] = league_sc.iloc[hIndex].teamPL - league_sc.iloc[aIndex].teamPL
    
    features = features.append(fea.T, ignore_index=True)
    
    features = HM_one_hot_encoder(features)
    
    features['gid'] = gid
    
    tfsys.samples = tfsys.samples.append(features)
 


#获取积分、进失球等数据
def get_score_data(htm, keyword):
    cols = ['HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 
        'HTGD', 'ATGD', 'DiffPts', 'DiffLP']
    score = pd.Series(index=cols,dtype=str)
    scoreDF = pd.DataFrame(columns=cols)
    
    home = re.findall(r"var h2h_home = .*", htm)
    home = zstr.str_flt(home[0], [';',' ','\n','\t','\r'])
    homeId = (re.split("\=", home))[1]

    away = re.findall(r"var h2h_away = .*", htm)
    away = (re.split("\=", away[0]))[1]
    awayId = zstr.str_flt(away, [';',' ','\n','\t','\r'])
    
    HTGS = 0.0
    HTGC = 0.0
    HTP = 0.0
    ATGS = 0.0
    ATGC = 0.0
    ATP = 0.0
    datas = re.findall(keyword, htm)
    datas = datas[0].split('\",\"')
    for data in datas:
        keystr = '|'
        pos = data.index(keystr)
        data = data[pos:]
        res = re.split("\|", data)
        teamId = res[2]
        if teamId==homeId:
            HomeTeamLP = float(res[1])
            HTGS = float(res[15])
            HTGC = float(res[16])
            HTP = float(res[17])
        if teamId==awayId:
            AwayTeamLP = float(res[1])
            ATGS = float(res[15])
            ATGC = float(res[16])
            ATP = float(res[17])
            
    HTGD = HTGS - HTGC
    ATGD = ATGS - ATGC
    DiffPts = HTP - ATP
    DiffLP = HomeTeamLP - AwayTeamLP
    
    score['HTGS'] = HTGS
    score['ATGS'] = ATGS
    score['HTGC'] = HTGC
    score['ATGC'] = ATGC
    score['HTP'] = HTP
    score['ATP'] = ATP
    score['HTGD'] = HTGD
    score['ATGD'] = ATGD
    score['DiffPts'] = DiffPts
    score['DiffLP'] = DiffLP
    
    scoreDF = scoreDF.append(score.T, ignore_index=True)

    return scoreDF




#获取对赛的结果 
def get_VS_result(htm, keyword):
    result = re.findall(keyword, htm)
    res = result[0].split(' = ')
    data = re.findall(r"\[\[(.*)\]\]", res[0])
    data = data[0].split('],[')
    TFormPtsStr = ''
    
    for game in data:
        res = re.split("\,", game)
        kwin = res[11]        
        if kwin == '1': 
            FTR = 'W'
        elif kwin == '0':
            FTR = 'D'
        elif kwin == '-1':
            FTR = 'L'
        else: FTR = 'M'
        TFormPtsStr = TFormPtsStr + FTR
     
    if len(TFormPtsStr) >= 5:
        TFormPtsStr = TFormPtsStr[:5] 
    else: 
        for i in range(len(TFormPtsStr)-5):TFormPtsStr = kwins+'M'


    return TFormPtsStr  
    
    
    
    

### 提取进失球、积分、排名、对赛、近几场赛果等特征
def fb_get_score_features(htm, bars, ftg=''):
    cols = ['VM1', 'VM2', 'VM3', 'VM4', 'VM5',
            'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 
            'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
            'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
            'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
            'DiffFormPts']
    matchDF = pd.DataFrame(columns=cols)
    match = pd.Series(index=cols,dtype=str)
    
    VTFormPtsStr = get_VS_result(htm, "var v_data=\[.*\]")
    HTFormPtsStr = get_VS_result(htm, "var h_data=\[.*\]")
    ATFormPtsStr = get_VS_result(htm, "var a_data=\[.*\]")
    scoreDF = get_score_data(htm, "var ScoreAll = Array(.*)")

    ##近5场对赛结果
    match['VM1'] = VTFormPtsStr[0]
    match['VM2'] = VTFormPtsStr[1]
    match['VM3'] = VTFormPtsStr[2]
    match['VM4'] = VTFormPtsStr[3]
    match['VM5'] = VTFormPtsStr[4]
    ##近5场主队比赛结果
    match['HM1'] = HTFormPtsStr[0]
    match['HM2'] = HTFormPtsStr[1]
    match['HM3'] = HTFormPtsStr[2]
    match['HM4'] = HTFormPtsStr[3]
    match['HM5'] = HTFormPtsStr[4]
    ##近5场客队比赛结果
    match['AM1'] = ATFormPtsStr[0]
    match['AM2'] = ATFormPtsStr[1]
    match['AM3'] = ATFormPtsStr[2]
    match['AM4'] = ATFormPtsStr[3]
    match['AM5'] = ATFormPtsStr[4]
    
    match['HTWinStreak3'] = get_3game_ws(HTFormPtsStr)
    match['HTWinStreak5'] = get_5game_ws(HTFormPtsStr)
    match['HTLossStreak3'] = get_3game_ls(HTFormPtsStr)
    match['HTLossStreak5'] = get_5game_ls(HTFormPtsStr)   
    match['ATWinStreak3'] = get_3game_ws(ATFormPtsStr)
    match['ATWinStreak5'] = get_5game_ws(ATFormPtsStr)
    match['ATLossStreak3'] = get_3game_ls(ATFormPtsStr)
    match['ATLossStreak5'] = get_5game_ls(ATFormPtsStr)
    
    HTFormPts = get_form_points(HTFormPtsStr)
    ATFormPts = get_form_points(ATFormPtsStr)  
    DiffFormPts = HTFormPts - ATFormPts
    match['DiffFormPts'] = DiffFormPts
    
    matchDF = matchDF.append(match.T, ignore_index=True)
    
    feature = pd.concat([scoreDF, matchDF], axis=1)
    
    kk=0
    
    '''
    result = re.findall(r"var v_data=\[.*\]", htm)
    res = result[0].split(' = ')
    v_data = re.findall(r"\[\[(.*)\]\]", res[0])
    v_data = v_data[0].split('],[')
    kwins = ''
    vcols = ['VM1','VM2','VM3','VM4','VM5']
    v_feature = pd.Series(index=vcols,dtype=str)
    
    for game in v_data:
        res = re.split("\,", game)
        qj = int(res[8]) 
        qs = int(res[9])        
        if qj > qs: 
            kwin = 'W'
        elif qj < qs:
            kwin = 'L'
        elif qj == qs:
            kwin = 'D'
        else: kwin = 'M'
        kwins = kwins + kwin
     
    if len(kwins) >= 5:
        kwins = kwins[:5] 
    else: 
        for i in range(len(kwins)-5):kwins = kwins+'M'

    ###1. 最近5场对赛赛果
    v_feature['VM1'] = kwins[0]
    v_feature['VM2'] = kwins[1]
    v_feature['VM3'] = kwins[2]
    v_feature['VM4'] = kwins[3]
    v_feature['VM5'] = kwins[4]
    ###2. 最近5场对赛主队积分
    point5 = get_form_points(kwins)

    
    result = re.findall(r"var h_data=\[.*\]", htm)
    res = result[0].split(' = ')
    h_data = re.findall(r"\[\[(.*)\]\]", res[0])
    h_data = h_data[0].split('],[')
    hcols = ['HM1','HM2','HM3','HM4','HM5']
    h_feature = pd.Series(index=hcols,dtype=str)
    
    for game in h_data:
        res = re.split("\,", game)

    '''
        
    
    return VTFormPtsStr






def fb_gid_getExt010(x10):
    bars=pd.Series(x10,index=tfsys.gidSgn,dtype=str)
    gid=bars['gid']
    isdownload = False
#    gid = '1549938'

    ### 1.下载投注量网页
    uss_tz=tfsys.us0_extTouzhu+ gid +'.htm?%s'
    fss_tz=tfsys.rhtmTouzhu + gid + '.htm'
    fxdat_tz=tfsys.rxdat + gid + '_tz.dat' 
    
    for timeout in range(5): #没有下载到网页，就重复5遍
        if not isdownload:
            #获取当前时间
            uss_tz = uss_tz % (time.mktime(datetime.datetime.now().timetuple()))
            htm_tz = zweb.web_get001txtFg(uss_tz,fss_tz) #zt.zt_web_get001txtFg or(fsiz<5000):
            if htm_tz=='404': #### 没有下载到网页
#                print('############ 404 ############')
                isdownload = False
            else: isdownload = True
    if not isdownload:
        return '######### can not download the html ##########'        
    df = fb_gid_getExt_tz4htm(htm_tz,bars,ftg=fxdat_tz)
    if df.empty: return
    
    ### 2.下载赔率网页  
    uss_oz = tfsys.us0_extOuzhi + gid + '.js'
    fss_oz = tfsys.rhtmOuzhi + gid + '.js'
    fxdat_oz_1=tfsys.rxdat + gid + '_oz_1.dat'
    fxdat_oz_2=tfsys.rxdat + gid + '_oz_2.dat'
    htm_oz = zweb.web_get001txtFg(uss_oz, fss_oz)    
    fb_gid_getExt_oz4htm_1(htm_oz, bars,ftg=fxdat_oz_1)
    
    ### 3. 下载分析网页
    uss_fx = tfsys.us0_extFenxi + gid + '.htm'
    fss_fx = tfsys.rhtmFenxi + gid + '.htm'
    fxdat_fx = tfsys.rxdat + gid + '_fx.dat'
    htm_fx = zweb.web_get001txtFg(uss_fx, fss_fx)   
    fb_get_score_features(htm_fx, bars,ftg=fxdat_fx)
    
#    fb_gid_getExt_oz4htm_1(htm_oz,bars,ftg=fxdat_oz_1)
#    fb_gid_getExt_oz4htm_2(htm_oz,bars,ftg=fxdat_oz_2)
    
    return fss_oz


def fb_gid_getExt(df):
    dn9=len(df['gid'])
    for i, row in df.iterrows():
        #xtfb.kgid=row['gid']
        #xtfb.bars=row
        fb_gid_getExt010(row.values)
        #
        print(zsys.sgnSP8,i,'/',dn9,'@ext')

    
def fb_gid_getExtPool(df,nsub=6):
    pool=ThreadPoolExecutor(max_workers = nsub)
    xsubs = [pool.submit(fb_gid_getExt010,x10) for x10 in df.values]
    #
    dn9=len(df['gid'])
    ns9=str(dn9)
    for xsub in as_completed(xsubs):
        fss=xsub.result(timeout=20);
        print('@_getExtPool,xn9:',ns9,fss)


def fb_get_team_dataset(htm):
    ### 1.获取轮赛参赛队伍
    teamDict = {}
    
    result = re.findall(r"var arrTeam = \[.*\]", htm)
    res = result[0].split(' = ')
    teams = re.findall(r"\[\[(.*)\]\]", res[1])
    teams = teams[0].split('],[')
    
    for team in teams:
        res = re.split("\,", team)
        team_id = res[0]
        team_name = res[1]
        team_name = zstr.str_flt(team_name, '\'')
        teamDict[team_id] = team_name

    ### 2. 获取参赛队伍积分，进失球等
    scoresDf = pd.DataFrame(columns=tfsys.scSgn,dtype=str)
    ds = pd.Series(tfsys.scNil,index=tfsys.scSgn,dtype=str)
    
    result = re.findall(r"var totalScore = \[.*\]", htm)
    res = result[0].split(' = ')
    scores = re.findall(r"\[\[(.*)\]\]", res[1])
    scores = scores[0].split('],[')
    
    for score in scores:
        res = re.split("\,", score)
        ds['teamPL'] =int(res[1])  #名次
        ds['team_id'] = res[2] #球队ID
        ds['MW'] = int(res[4])   #轮次
        ds['wins'] = int(res[5])   #胜场数
        ds['draws'] = int(res[6])  #平
        ds['losts'] = int(res[7])  #负
        ds['goal_score'] = float(res[8])   #进球数
        ds['goal_conceded'] = float(res[9])  #失球数
        ds['goal_diff'] = float(res[10])    #净胜球数
        ds['GS'] = float(res[14]) #场均进球数
        ds['GC'] = float(res[15]) #场均失球数
        ds['TP'] = float(res[16]) / int(res[4]) #场均得分
        ds['M1'] = get_sign(res[24]) #前 1 轮赛果
        ds['M2'] = get_sign(res[23]) #前 2 轮赛果
        ds['M3'] = get_sign(res[22]) #前 3 轮赛果
        ds['M4'] = get_sign(res[21]) #前 4 轮赛果
        ds['M5'] = get_sign(res[20]) #前 5 轮赛果
        scoresDf = scoresDf.append(ds.T,ignore_index=True)
    
    gs_max = (scoresDf['goal_score']).max()
    gc_max = (scoresDf['goal_conceded']).max()
    scoresDf['goal_score'] = scoresDf['goal_score'] / gs_max
    scoresDf['goal_conceded'] = scoresDf['goal_conceded'] / gc_max

    return teamDict, scoresDf


#---- download samples
# 下载轮赛每球队的数据，fgSample为下载赛季每场比赛的数据的标志 
def fb_download_league_data(league, fgSample=False):
    leagueId = tfsys.leagueId[league]
    subleagueId = tfsys.subleagueId[league]
    season = '2018-2019'

    fss=tfsys.lghtm + season + '_' + leagueId + '.js'        
    uss=tfsys.us0_league + season + '/s' + leagueId + subleagueId + '.js'      

    htm=zweb.web_get001txtFg(uss,fss)
#    if len(htm)>5000:

    ### 1. 轮赛每队当前的基本面，包括进、失球，积分，排名等
    teamDict, scoresDf = fb_get_team_dataset(htm)
#    tfsys.teamIds = tfsys.teamIds.append(teamDf)
    tfsys.teamIds = dict(tfsys.teamIds, **teamDict)
#    tfsys.teamIds.drop_duplicates(subset='team_id', keep='last', inplace=True)


    ### 2. 过去赛季比赛场次ID，赔率、及当时两队的对赛记录（用于算法的训练）
    if fgSample: fb_league_gids(htm, league)
#    
#        if fgExt:fb_gid_getExt(df)
#        else: fb_gid_getExtPool(df)
#
#    if tfsys.gidsFN!='':
#        print('+++++')
#        print(tfsys.gids.tail())
#        tfsys.gids.to_csv(tfsys.gidsFN,index=False,encoding='gb18030')
        
    return scoresDf 



def fb_gid_get_nday(xtfb,timStr,fgExt=False):
    ###1. 下载各轮赛数据
    for league in tfsys.league:
        sc = fb_download_league_data(league)
        tfsys.league_sc = pd.concat([tfsys.league_sc, sc], ignore_index=True)
        
    ### 2.下载当期比赛
    if timStr=='':ktim=xtfb.tim_now
    else:ktim=arrow.get(timStr)
    #
    nday=tfsys.xnday_down
    for tc in range(nday):
        xtim=ktim.shift(days=-tc)
        xtimStr=xtim.format('YYYY-MM-DD')
        #print('\nxtim',xtim,xtim<xtfb.tim0_gid)
        #
        xss=str(tc)+'#,'+xtimStr+',@'+ zt.get_fun_nam()
        zt.f_addLog(xss)
        if xtim<xtfb.tim0_gid:
            print('#brk;')
            break
        #       
        fss=tfsys.rghtm+xtimStr+'.htm'          
        uss=tfsys.us0_gid+xtimStr
        
        print(timStr,tc,'#',fss)
        #
        htm=zweb.web_get001txtFg(uss,fss)
        if len(htm)>5000:
            df=fb_gid_get4htm(htm)  #提取每天比赛场次
            if len(df['gid'])>0:
                tfsys.gids=tfsys.gids.append(df)
                tfsys.gids.drop_duplicates(subset='gid', keep='last', inplace=True)
                #
                if fgExt:fb_gid_getExt(df)        #单线程
                #if fgExt:fb_gid_getExtPool(df)   #多线程 
    #
    if tfsys.gidsFN!='':
        print('+++++')
        print(tfsys.gids.tail())
        tfsys.gids.to_csv(tfsys.gidsFN,index=False,encoding='gb18030')
    
