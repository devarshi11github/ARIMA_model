#time series forecasting using ARIMA model - for the problem staement 'Predict Future Sales' https://www.kaggle.com/c/competitive-data-science-predict-future-sales

#https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts

#changing the directory and getting the list in a directory
import os
os.chdir("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales")
dirlist = os.listdir()
dirlist
#Basic packages
import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt #basic plotting
import seaborn as sns
#Time series
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller,acf,pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodel.api as sm
import scipy.stats as scs
#settings
import warnings
warnings.filterwarnings("ignore")
#import all of them
sales = pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/sales_train.csv/sales_train_v2.csv")
#settings
import warnings
warnings.filterwarnings("ignore")
item_cat=pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/item_categories.csv")
item=pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/items.csv")
sub=pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/sample_submission.csv/sample_submission.csv")
shops=pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/shops.csv")
test=pd.read_csv("C:/Users/dmandal/Desktop/Devarshi HP data/OTHER DESKTOP FILES/Data Analysis/Python/Predict_Future_Sales/Predict_future_sales/test.csv/test.csv")
#formatting the date column in the sales dataset correctly
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
#checking the data
print(sales.info())
#aggregate to mothly level the required metrics
monthly_sales = sales.groupby(["date_block_num","shop_id","item_id"])["date","item_price","item_cnt_day"].agg({"date":["min","max"],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head()
#number of items per cat
x = item.groupby(['item_category_id']).count()
x = x.sort_values(by='item_id',ascending=False)
x = x.iloc[0:10].reset_index()
x
#plot
plt.figure(figsize=(8,4))
ax=sns.barplot(x.item_category_id,x.item_id,alpha=0.8)
plt.title("Items per category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category',fontsize=12)
plt.show()
#predict sales for next month at a store item combination
#total sales per month and plot data, also check groupby weekly basis
ts = sales.groupby(['date_block_num'])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);
#getting the rolling mean and rolling sd
plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();
#there is a seasonality and decreasing trend
#decomposition to trend, seasonality and residuals
import statsmodels.api as sm
#multiplicative, decomposing it into trend, seasonality and residual
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model='multiplicative')
#plt.figure(figsize=(16,12))
fig=res.plot()
fig.show()
#The additive model is useful when the seasonal variation is relatively constant over time. The multiplicative model is useful when 
#the seasonal variation increases over time.
#Additive model
res=sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsizze=(16,12))
fig=res.plot()
#fig.show()
#Additive model: yt=St+Tt+Et, Multiplicative model: yt=St x Tt x Et
#a stationary series allows usageof model (mean should not be a function of time, variance should not be a function-called homoscedasticity), 
#covariance of ith term and the (i+m)th term should not be a function of time
#when time series is stationary it is easier to model
#multiple checks to test stationarity, ADF (Augmented Dicky fuller test), KPSS, PP (Philips-Perron test)
#stationarity tests
def test_stationarity(timeseries):
    
    #perform Dickey-Fuller test:
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value    
    print(dfoutput)

test_stationarity(ts)

#we see that the p value (0.142953) is greater than .05 and test statistic (-2.395704) is greater than critical values, hence no need to reject null hypothesis
# hence time series is non stationary
#to remove trend
from pandas import Series as Series
#create a differenced series
def difference(dataset,interval=1):
    diff=list()
    for i in  range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


ts= sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)  #assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()
#now testing the stationarity again after de-seasonality
test_stationarity(new_ts)
#p value is less than .05 hence the null hypothesis can be rejected, hence stationarity of series
#AR, MA, or ARMA model
def tsplot(y, lags=None, figsize=(10,8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y=pd.Series(y)












