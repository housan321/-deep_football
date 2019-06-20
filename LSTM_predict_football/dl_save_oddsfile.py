# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:53:49 2018

@author: Administrator
"""


import dl_data_process as dp


odds_file_name = 'odds_4_cnn(1.5).dat'
odds_file_name_svm = 'odds_4_svm(1.5).dat'
odds_file_name_xgboost = 'data/odds_4_xgboost(5.26).dat'

#dp.save_odds_to_file(odds_file_name, [0, 20000])
#dp.save_odds_to_file_for_svm(odds_file_name_svm, [0, 20000])
dp.save_odds_to_file_for_xgboost(odds_file_name_xgboost, [0, 20000])

print("save odds to file OK")

'''
# kwin, data = dp.load_odds_file2(['f_odds(more)1.dat', 'f_odds(more)2.dat', 'f_odds(more)3.dat'])

files = ['f_odds(more)1.dat', 'f_odds(more)2.dat', 'f_odds(more)3.dat']
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = dp.load_data2(files)
'''
