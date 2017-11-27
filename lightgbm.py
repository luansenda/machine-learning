# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:12:04 2017

@author: Administrator
"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# load or create your dataset
print('Load data...')
df_train = pd.read_csv(r'E:\py_data\xgb_train.csv')
df_test = pd.read_csv(r'E:\py_data\xgb_test.csv')

y_train = df_train['count'].values
y_test = df_test['count'].values
X_train = df_train.drop('count', axis=1).values #extract first column from data source and save as dependent variable
X_test = df_test.drop('count', axis=1).values

print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=100,
                        learning_rate=0.1,
                        n_estimators=40)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

print('Plot feature importances...')
ax = lgb.plot_importance(gbm)
plt.show()

## other scikit-learn modules  for params optimazition
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)





