import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp
from random import sample
df = pd.read_csv('GEFCom2014.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# for i in df.columns:
#     print(i, ': ', df[i].unique())
#
# print(df['Load'].isna().sum())  # 35064
ld = df['Load'].tolist()
ld_7 = []

for i in range(len(ld)):
    if i >= 1:
        ld_7.append(ld[i-1])
    else:
        ld_7.append(np.mean(ld[:1]))


df['load_h1'] = pd.DataFrame(ld_7)

#ld1 = df['Load'].tolist()
ld_8 = []

for i in range(len(ld)):
    if i >= 168:
        ld_8.append(ld[i - 168])
    else:
        ld_8.append(np.mean(ld[:168]))

df['load_d7'] = pd.DataFrame(ld_8)


ld2 = df['Load'].tolist()
ld_9 = []

for i in range(len(ld)):
    if i >= 2:
        ld_9.append(ld[i-2])
    else:
        ld_9.append(np.mean(ld[:2]))


df['load_h2'] = pd.DataFrame(ld_9)


ld = df['Load'].tolist()
ld_10 = []

for i in range(len(ld)):
    if i >= 3:
        ld_10.append(ld[i-3])
    else:
        ld_10.append(np.mean(ld[:3]))


df['load_h3'] = pd.DataFrame(ld_10)

ld = df['Load'].tolist()
ld_11 = []

for i in range(len(ld)):
    if i >= 4:
        ld_11.append(ld[i-4])
    else:
        ld_11.append(np.mean(ld[:4]))


df['load_h4'] = pd.DataFrame(ld_11)



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


# inp = X_train.shape[1]
# inputs = tf.keras.Input(shape=(inp,))
# h1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
# h2 = tf.keras.layers.Dense(4, activation=tf.nn.relu)(h1)
# outputs = tf.keras.layers.Dense(1)(h2)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# # model.add(tf.keras.layers.Dense(train.shape[1], activation='relu'))
# # model.add(tf.keras.layers.Dense(16, activation='relu'))
# # model.add(tf.keras.layers.Dense(4, activation='relu'))
# # model.add(tf.keras.layers.Dense(1))
#
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-2))  # keras.optimizers.Adam(learning_rate=1e-5)
#
#
# model.fit(X_train.values, Y_train.values, epochs=100, shuffle=True, verbose=2)
#
# y_pred = model.predict(X_test)
#
# test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
#
# df1 = pd.DataFrame(y_pred)
#
# y_pred = y_pred.tolist()
# y_val = Y_test['Load'].tolist()
#
#
# def r2_keras(y_true, y_pred):
#     SS_res =  np.sum(np.square(y_true - y_pred))
#     SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
#     return  1 - SS_res/(SS_tot + tf.keras.backend.epsilon())
#
#
# print('r2 score: ', r2_keras(y_val, y_pred))
#
# print(df1)
# print('Y_TEST:\n', Y_test)
# print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
#
#


import lightgbm as lgb
from sklearn.metrics import mean_squared_error

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression',
              'metric': {'rmse'}, 'num_leaves': 8, 'learning_rate': 0.05,
              'feature_fraction': 0.8, 'max_depth': 7, 'verbose': 0,
              'num_boost_round':25000, #'early_stopping_rounds':100,
           'nthread':-1}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval)
                #early_stopping_rounds=5)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(Y_test, y_pred) ** 0.5)

#
df_test = pd.get_dummies(test)
df_test.drop('Load',axis=1,inplace=True)
X_prediction = df_test.values
#
predictions = gbm.predict(X_prediction,num_iteration=gbm.best_iteration)

sub = test.loc[:,['Load']]
sub['sales']= predictions
sub['Day']=test.loc[:,['Day']]
sub['Month']=test.loc[:,['Month']]
sub['Year']=test.loc[:,['Year']]
sub['Accuracy']= 100 - (abs(sub['sales']-sub['Load'])/sub['Load'])*100
print(sub.describe())
df3 = pd.read_excel('1.xls')
print(df3.head())
df3 = sub
df3.to_excel('1.xls')
