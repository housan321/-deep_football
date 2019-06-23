# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 

  
文件名:tfb_sys.py
默认缩写：import tfb_sys as tfsys
简介：Top极宽量化·足彩系统参数模块
 

'''
#

import sys,os,re
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

#
import numexpr as ne  

#
import zsys
import ztools as zt
import zpd_talib as zta



#-----------global.var&const
#gidDType={'gid':str,'jq':int,'sq':int,'rq':int,'kend':int}
gidNil=['','','','-1','','-1',  '-1','-1','0',  '0','-1','-1',  '','','']
gidSgn=['gid','gset','mplay','mtid','gplay','gtid', 'qj','qs','qr',  'kend','kwin','kwinrq', 'tweek','tplay','tsell']
#gidNil14=['','','','','','','']
#gidSgn14=['dateid','gid','gset','mplay','gplay', 'tplay','tsell']
#
poolNil=['','','','','','',  '-1','-1','0',  '0','-1','-1',  '','','', '0',0,0,0, '-9']
poolSgn=['gid','gset','mplay','mtid','gplay','gtid', 'qj','qs','qr',  'kend','kwin','kwinrq', 'tweek','tplay','tsell'
         ,'cid','pwin9','pdraw9','plost9'  , 'kwin_sta']
#

###亚盘参数
###################################################################
gxdatNil_az=['','','',  0,0,0,0,0,0, 
         '','','','','', '-1','-1','0','-1','-1', '','' ]

gxdatSgn_az=['gid','cid','cname',
  'mshui0','pan0','gshui0','mshui9','pan9','gshui9',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']

pan = {'平手': '0',
       '平手/半球': '0.25',
       '半球': '0.5',
       '半球/一球': '0.75',
       '一球': '1.0',
       '一球/球半': '1.25',
       '球半': '1.5',
       '球半/两球': '1.75',
       '两球': '2.0',
       '两球/两球半': '2.25',
       '两球半': '2.5',
       '两球半/三球':'2.75',
       '三球': '3.0',
       '三球/三球半':'3.25',
       '三球半':'3.5',
       '三球半/四球':'3.75',
       '四球':'4.0',
       '四球/四球半':'4.25',
       '四球半':'4.5',
       '四球半/五球':'4.75',
       '五球':'5.0', 
       '五球/五球半':'5.25',
       '五球半':'5.5',
       '五球半/六球':'5.75',
       '六球':'6.0',
       '六球/六球半':'6.25',
       '六球半':'6.5',
       '六球半/七球':'6.75',
       '七球':'7.0',
       '七球/七球半':'7.25',
       '七球半':'7.5',
       '七球半/八球':'7.75',
       '八球':'8.0',
       
       '受平手/半球': '-0.25',
       '受半球': '-0.5',
       '受半球/一球': '-0.75',
       '受一球': '-1.0',
       '受一球/球半': '-1.25',
       '受球半': '-1.5',
       '受球半/两球': '-1.75',
       '受两球': '-2.0',
       '受两球/两球半': '-2.25',
       '受两球半': '-2.5',
       '受两球半/三球':'-2.75',
       '受三球': '-3.0',
       '受三球/三球半':'-3.25',
       '受三球半':'-3.5',
       '受三球半/四球':'-3.75',
       '受四球':'-4.0',
       '受四球/四球半':'-4.25',
       '受四球半':'-4.5',
       '受四球半/五球':'-4.75',
       '受五球':'-5.0',
       '受五球/五球半':'-5.25',
       '受五球半':'-5.5',
       '受五球半/六球':'-5.75',
       '受六球':'-6.0',
       '受六球/六球半':'-6.25',
       '受六球半':'6.5',
       '受六球半/七球':'-6.75',
       '受七球':'-7.0',
       '受七球/七球半':'-7.25',
       '受七球半':'-7.5',
       '受七球半/八球':'-7.75',
       '受八球':'-8.0',}  

az_pan = {'平手': '0',
       '平/半': '0.25',
       '半球': '0.5',
       '半/一': '0.75',
       '一球': '1.0',
       '一/球半': '1.25',
       '球半': '1.5',
       '球半/两': '1.75',
       '两球': '2.0',
       '两/两球半': '2.25',
       '两球半': '2.5',
       '两球半/三':'2.75',
       '三球': '3.0',
       '三/三球半':'3.25',
       '三球半':'3.5',
       '三球半/四':'3.75',
       '四球':'4.0',
       '四/四球半':'4.25',
       '四球半':'4.5',
       '四球半/五':'4.75',
       '五球':'5.0', 
       '五/五球半':'5.25',
       '五球半':'5.5',
       '五球半/六':'5.75',
       '六球':'6.0',
       '六/六球半':'6.25',
       '六球半':'6.5',
       '六球半/七':'6.75',
       '七球':'7.0',
       '七/七球半':'7.25',
       '七球半':'7.5',
       '七球半/八':'7.75',
       '八球':'8.0',
       
       '受平手': '0',
       '受平/半': '-0.25',
       '受半球': '-0.5',
       '受半/一': '-0.75',
       '受一球': '-1.0',
       '受一/球半': '-1.25',
       '受球半': '-1.5',
       '受球半/两': '-1.75',
       '受两球': '-2.0',
       '受两/两球半': '-2.25',
       '受两球半': '-2.5',
       '受两球半/三':'-2.75',
       '受三球': '-3.0',
       '受三/三球半':'-3.25',
       '受三球半':'-3.5',
       '受三球半/四':'-3.75',
       '受四球':'-4.0',
       '受四/四球半':'-4.25',
       '受四球半':'-4.5',
       '受四球半/五':'-4.75',
       '受五球':'-5.0',
       '受五/五球半':'-5.25',
       '受五球半':'-5.5',
       '受五球半/六':'-5.75',
       '受六球':'-6.0',
       '受六/六球半':'-6.25',
       '受六球半':'6.5',
       '受六球半/七':'-6.75',
       '受七球':'-7.0',
       '受七/七球半':'-7.25',
       '受七球半':'-7.5',
       '受七球半/八':'-7.75',
       '受八球':'-8.0',}  

cidrows= 20

delSgn_oz=['cname',
  'vwin0','vdraw0','vlost0','vwin9','vdraw9','vlost9',
  'vback0','vback9',
  'vwin0kali','vdraw0kali','vlost0kali','vwin9kali','vdraw9kali','vlost9kali',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwinrq',  
  'tweek','tplay']


delSgn_az=['gid','cid','cname',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']

###################################################################
###################################################################

###欧盘参数
###################################################################
gxdatNil=['','','',  0,0,0,0,0,0,  0,0,0,0,0,0, 0,0, 0,0,0,0,0,0,
         '','','','','', '-1','-1','0','-1','-1', '','' ]
gxdatSgn=['gid','cid','cname',
  'pwin0','pdraw0','plost0','pwin9','pdraw9','plost9',
  'vwin0','vdraw0','vlost0','vwin9','vdraw9','vlost9',
  'vback0','vback9',
  'vwin0kali','vdraw0kali','vlost0kali','vwin9kali','vdraw9kali','vlost9kali',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']
###################################################################

### 主流菠菜公司ID
cids=['281','115','82','173','81','90','104','352','16','18','976','255','88',
      '545','70','158','97','4','370','177','110','60','450','80','422','499',
      '474','517','659','2','841','665','601','1047']


###投注参数
###################################################################
gxdatNil_tz=[
            0,0,0, 
            0,0,0, 
            0,
            0,0,0, 
            0,0,0,
            0,
            0,0,0, 
            0,0,0, 
            0,0,0,
            0,0,0,
            0,0,0, 
            0,0,0]
gxdatSgn_tz=[
  'av_odd_h','av_odd_d','av_odd_g',
  'av_prob_h','av_prob_d','av_prob_g',
  'av_back',
  'bf_odd_h','bf_odd_d','bf_odd_g',
  'bf_prob_h','bf_prob_d','bf_prob_g',
  'bf_back',
  'volume_h','volume_d','volume_g',
  'vol_ratio_h','vol_ratio_d','vol_ratio_g',
  'shui_h','pan','shui_g',
  'profit_h','profit_d','profit_g',
  'pf_idx_h','pf_idx_d','pf_idx_g',
  'hot_idx_h','hot_idx_d','hot_idx_g']


  
#
retNil=['', 0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
retSgn=['xtim', 'kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','ret1','ret0',  'nwin3','nwin1','nwin0',  'num3','num1','num0']
#retNil=[0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
#retSgn=['kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','num3','nwin3', 'ret1','num1','nwin1', 'ret0','num0','nwin0']

#--bt.var  
btvarNil=['', 0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,   0,0,0, 0,0,0,'']
btvarSgn=['xtim', 'kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','ret1','ret0',  'nwin3','nwin1','nwin0',  'num3','num1','num0'
          ,'v1','v2','v3','v4','v5','nday','doc']

league=['瑞典超', '挪超', 
        '日职联', '日职乙', '韩K联', 
        '美职业', '巴西甲']
leagueId={'英超':'36', '西甲':'31', '意甲':'34', '德甲':'8', '法甲':'11', 
          '英冠':'37', '挪超':'22', '瑞典超':'26',
          '日职联':'25', '日职乙':'284', '韩K联':'15', 
          '美职业':'21', '巴西甲':'4'}
subleagueId={'英超':'', '西甲':'', '意甲':'', '德甲':'', '法甲':'',
             '英冠':'_87', '挪超':'', '瑞典超':'_431', 
             '日职联':'_943', '日职乙':'_808', '韩K联':'_313', 
             '美职业':'_165', '巴西甲':''}
 
scNil = [0, '-1', 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 
        'M', 'M', 'M', 'M', 'M']
scSgn = ['teamPL', 'team_id', 'MW', 'wins', 'draws', 'losts',
         'goal_score', 'goal_conceded', 'goal_diff', 'GS', 'GC', 'TP', 
         'M1', 'M2', 'M3', 'M4', 'M5']

league_sc = pd.DataFrame(columns=scSgn,dtype=str) 
#self.nsum,self.nwin,self.ndraw,self.nlost=0,0,0,0
#self.kwin,self.kdraw,self.klost=0,0,0
#-------------------
#
#us0='http://trade.500.com/jczq/?date='
#http://odds.500.com/fenxi/shuju-278181.shtml
#http://odds.500.com/fenxi/yazhi-278181.shtml
#http://odds.500.com/fenxi/ouzhi-278181.shtml

us14_gid='http://trade.500.com/rcjc/'
#us0_gid='http://trade.500.com/jczq/?date='
#us0_ext0='http://odds.500.com/fenxi/'
us0_league='http://info.haocai138.com/jsData/matchResult/'
us0_gid='http://a.haocai138.com/buy/JingCai.aspx?typeID=101&oddstype=2&date='
us0_ext0='http://1x2d.win007.com/'
us0_extOuzhi=us0_ext0
us0_extYazhi=us0_ext0+'yazhi-'
us0_extFenxi='http://a.haocai138.com/analysis/'
us0_extTouzhu='http://a.haocai138.com/analysis/odds/'
#
rdat0='/tfbDat/'
rxdat=rdat0+'xdat/'
rmdat=rdat0+'mdat/'
rmlib=rdat0+'mlib/' #ai.mx.lib.xxx

#rgdat=rdat0+'gdat/'
#
lghtm=rdat0+'xhtm/league/'  #联赛网页
rghtm=rdat0+'xhtm/ghtm/'  #gids_htm,days 当天比赛场次
rhtmOuzhi=rdat0+'xhtm/htm_oz/'
rhtmYazhi=rdat0+'xhtm/htm_az/'
rhtmFenxi=rdat0+'xhtm/htm_fx/'
rhtmTouzhu=rdat0+'xhtm/htm_tz/'
#        

#---glibal.lib.xxx
gids=pd.DataFrame(columns=gidSgn,dtype=str)
xdats=pd.DataFrame(columns=gxdatSgn,dtype=str)

#teamIds=pd.DataFrame()
teamIds={}
samples=pd.DataFrame()

gidsFN=''
gidsNum=len(gids.index)
xdatsNum=len(xdats.index)
#
xbars=None
xnday_down=0

#----------class.fbt

class zTopFoolball(object):
    ''' 
    设置TopFoolball项目的各个全局参数
    尽量做到all in one

    '''

    def __init__(self):  
        #----rss.dir
        
        #
        self.tim0Str_gid='2010-01-01'
        self.tim0_gid=arrow.get(self.tim0Str_gid)
        
        #
        self.gid_tim0str,self.gid_tim9str='',''
        self.gid_nday,self.gid_nday_tim9=0,0
        #
        self.tim0,self.tim9,self.tim_now=None,None,None
        self.tim0Str,self.tim9Str,self.timStr_now='','',''
        #
        
        self.kgid=''
        self.kcid=''
        self.ktimStr=''
        #
        #----pool.1day
        self.poolInx=[]
        self.poolDay=pd.DataFrame(columns=poolSgn)
        #----pool.all
        self.poolTrd=pd.DataFrame(columns=poolSgn)
        self.poolRet=pd.DataFrame(columns=retSgn)
        self.poolTrdFN,self.poolRetFN='',''
        #
        self.bars=None
        self.gid10=None
        self.xdat10=None
        
        #
        #--backtest.var
        self.funPre,self.funSta=None,None #funPre()是预处理数，funSta()是分类函数
        self.preVars,self.staVars=[],[]
        #--backtest.ai.var
        #
        self.ai_mxFN0=''
        self.ai_mx_sgn_lst=[]
        self.ai_xlst=[]
        self.ai_ysgn=''
        self.ai_xdat,self.ai_xdat=None,None
        
        #
        #
        
        #
        #--ret.var
        self.ret_nday,self.ret_nWin=0,0
        self.ret_nplay,self.ret_nplayWin=0,0
        
        self.ret_msum=0
        
        

#----------zTopFoolball.init.obj
        

    