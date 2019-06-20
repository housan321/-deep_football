# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 11:38:26 2018
@author: ASUS
"""

from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import numpy as np

import dl_data_process as dp




file =  'odds_4_svm(1.5).dat'
#files = ['odds2018(new2).dat']
# The data, split between train and test sets.
(x_train, y_train), (x_test, y_test) = dp.load_data_svm(file)

model = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, \
            shrinking=True, probability=True, tol=0.001, cache_size=2000,\
            class_weight=None, verbose=False, max_iter=-1, \
            decision_function_shape='ovo',random_state=None)
model.fit(x_train, y_train)

y_cls = model.predict(x_test)
y_pred = model.predict_proba(x_test)
#y_log_pred = model.predict_log_proba(x_test)
classif_rate = np.mean(y_cls.ravel() == y_test.ravel()) * 100
print("classif_rate for %s : %f " % ('svm', classif_rate))

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'svm_model2.m'
# Save model and weights
if not os.path.isdir(save_dir):
     os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
joblib.dump(model, model_path)
print('Saved trained model at %s ' % model_path)