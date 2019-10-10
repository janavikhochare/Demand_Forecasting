import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns

from fastai.imports import *
#from fastai.structured import *
from fbprophet import Prophet

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

import statsmodels.api as sm
# Initialize plotly
init_notebook_mode(connected=True)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

pd.option_context("display.max_rows", 1000)
pd.option_context("display.max_columns", 1000)
os.getcwd()
PATH = '/home/janavi/Desktop/demand_forcast'
print(os.listdir(PATH))
df_raw = pd.read_csv('/home/janavi/Desktop/demand_forcast/train.csv', low_memory=False, parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv('/home/janavi/Desktop/demand_forcast/test.csv', low_memory=False, parse_dates=['date'], index_col=['date'])
subs = pd.read_csv('/home/janavi/Desktop/demand_forcast/sample_submission.csv')


df_raw.head()

print("Train and Test shape are {} and {} respectively".format(df_raw.shape, df_test.shape))
#### Seasonality Check
# preparation: input should be float type
df_raw['sales'] = df_raw['sales'] * 1.0

# store types
sales_a = df_raw[df_raw.store == 2]['sales'].sort_index(ascending = True)
sales_b = df_raw[df_raw.store == 3]['sales'].sort_index(ascending = True) # solve the reverse order
sales_c = df_raw[df_raw.store == 1]['sales'].sort_index(ascending = True)
sales_d = df_raw[df_raw.store == 4]['sales'].sort_index(ascending = True)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))
c = '#386B7F'

# store types
sales_a.resample('W').sum().plot(color = c, ax = ax1)
sales_b.resample('W').sum().plot(color = c, ax = ax2)
sales_c.resample('W').sum().plot(color = c, ax = ax3)
sales_d.resample('W').sum().plot(color = c, ax = ax4)

#All Stores have same trend... Weird Seems like the dataset is A Synthetic One..;
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# Yearly
decomposition_a = sm.tsa.seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = sm.tsa.seasonal_decompose(sales_b, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = sm.tsa.seasonal_decompose(sales_c, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)

decomposition_d = sm.tsa.seasonal_decompose(sales_d, model = 'additive', freq = 365)
decomposition_d.trend.plot(color = c, ax = ax4)
date_sales = df_raw.drop(['store','item'], axis=1).copy() #it's a temporary DataFrame.. Original is Still intact..

date_sales.get_ftype_counts()
y = date_sales['sales'].resample('MS').mean()
y['2017':] #sneak peak
y.plot(figsize=(15, 6),)


decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
decomposition.plot()