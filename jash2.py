import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('GEFCom2014.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# for i in df.columns:
#     print(i, ': ', df[i].unique())
#
# print(df['Load'].isna().sum())  # 35064
ld = df['Load'].tolist()
ld_7 = []

for i in range(len(ld)):
    if i >= 7:
        ld_7.append(ld[i-7])
    else:
        ld_7.append(np.mean(ld[:7]))

df['load_d7'] = pd.DataFrame(ld_7)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.weekofyear
df = df.drop(['Date'], axis=1)

# print(df.columns)
# print(df.info())


df = df[np.isfinite(df['Load'])]

is_not_2011 = df['Year'] != 2011
train = df[is_not_2011]

is_2011 = df['Year'] == 2011
test = df[is_2011]

print(df.shape)
print(train.shape)
print(test.shape)

# print(train.head())
# print(test.head())
#
# print(train['Year'].unique())
# print(test['Year'].unique())


# model = tf.keras.Sequential()


X_train = pd.DataFrame(train.drop(['Load'], axis=1))
Y_train = pd.DataFrame(train['Load'])

X_test = pd.DataFrame(test.drop(['Load'], axis=1))
Y_test = pd.DataFrame(test['Load'])

print('train:\n', train.head())
print('test:\n', test.head())
print('X_train:\n', X_train.head())
print('Y_train:\n', Y_train.head())
print('X_test:\n', X_test.head())
print('Y_test:\n', Y_test.head())

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

corr =numeric_features.corr()
print(corr['Load'].sort_values(ascending=False))
#print(corr)
#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
plt.show()


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

train_data=lgb.Dataset(X_train, label=Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=train_data)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 15,
'num_boost_round':15000,
          'learning_rate': 0.07,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2,
          'reg_lambda': 1.2,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'
          }

# Create classifier to use
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 5,
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# View the default model params:
mdl.get_params().keys()

# # Create the grid
# grid = GridSearchCV(mdl, gridParams, verbose=2, cv=4, n_jobs=-1)
#
# # Run the grid
# grid.fit(X_train, Y_train)
#
# # Using parameters already set above, replace in the best from the grid search
# params['colsample_bytree'] = grid.best_params_['colsample_bytree']
# params['learning_rate'] = grid.best_params_['learning_rate']
# # params['max_bin'] = grid.best_params_['max_bin']
# params['num_leaves'] = grid.best_params_['num_leaves']
# #params['reg_alpha'] = grid.best_params_['reg_alpha']
# #params['reg_lambda'] = grid.best_params_['reg_lambda']
# params['subsample'] = grid.best_params_['subsample']
# # params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']
#
# print('Fitting with params: ')
# print(params)

lgbm = lgb.train(params,
                 train_data,
                  num_boost_round=20,
                 valid_sets=lgb_eval,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(Y_test, y_pred) ** 0.5)

#
df_test = pd.get_dummies(test)
df_test.drop('Load',axis=1,inplace=True)
X_prediction = df_test.values
#
predictions = lgbm.predict(X_prediction,num_iteration=lgbm.best_iteration)

sub = test.loc[:,['Load']]
sub['sales']= predictions
sub['Day']=test.loc[:,['Day']]
sub['Month']=test.loc[:,['Month']]
sub['Year']=test.loc[:,['Year']]
sub['Accuracy']= 100 - (abs(sub['sales']-sub['Load'])/sub['Load'])*100
print(sub.describe())
df3 = pd.read_excel('2.xls')
print(df3.head())
df3 = sub
df3.to_excel('2.xls')
