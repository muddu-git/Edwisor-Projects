data_encode2=data_pre
data_encode2["YearMonth"]=data_pre["Year"].astype(int).astype(str)+"-"+data_pre["Month of absence"].astype(int).astype(str)
data_encode2["YearMonth"]=pd.to_datetime(data_encode2["YearMonth"],format="%Y-%m")
data_encode2["YearMonth"]=data_encode2["YearMonth"].dt.strftime('%Y-%m')
data_encode2.index=data_encode2["YearMonth"]
ts=data_encode2["Absenteeism time in hours"]

import matplotlib.pyplot as plt
plt.plot(ts,label="Absent time in hours")
plt.title("time in hours")
plt.xlabel("Time(year-month)")
plt.ylabel("Absenteesim time in hours")
plt.legend(loc='best')

ts.to_csv("C:/Users/mudmoham/Desktop/timeseries.csv")

train_ts=ts.ix["2007-07":"2009-12"]
test_ts=ts.ix["2010-01":]

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries,window=5)
    rolstd = pd.rolling_std(timeseries,window=5)
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
	
	
test_stationarity(data_encode2['Absenteeism time in hours'])

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(ts)

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(ts)



from statsmodels.tsa.arima_model import ARIMA
import warnings
import itertools
warnings.filterwarnings("ignore") # specify to ignore warning messages
import statsmodels.api as sm 
p=range(3)
d=range(1)
q=range(3)
pdq=list(itertools.product(p,d,q))

train_ts_df=pd.DataFrame(train_ts)
test_ts_df=pd.DataFrame(test_ts)
train_list=[x for x in train_ts_df["Absenteeism time in hours"]]
test_list=[x for x in test_ts_df["Absenteeism time in hours"]]

min_aic_list=[]
for param in pdq:
	try:
		model=ARIMA(train_list,order=param)
		results = model.fit()
		min_aic_list.append([results.aic,param])
	except:
		continue
	            
min_aic_df=pd.DataFrame(min_aic_list,columns=["aic","param"])



model=ARIMA(train_ts,order=(0,0,2))
model_fit=model.fit()

print(model_fit.summary())
residuals_df=pd.DataFrame(model_fit.resid)
residuals_df.plot()
residuals_df.plot(kind='kde')
residuals_df.describe()


predictions=[]
for i in range(len(test_list)):
	model=ARIMA(train_list,order=(0,0,2))
	model_fit=model.fit()
	output=model_fit.forecast()
	p=output[0]
	predictions.append(p)
	obs=test_list[i]
	train_list.append(obs)
	
	
from sklearn.metrics import mean_squared_error,r2_score
print("MSE:{}".format(mean_squared_error(test_list,predictions)))
print("RMSE:{}".format(np.sqrt(mean_squared_error(test_list,predictions))))
print("r2 score:{}".format(r2_score(test_list,predictions)))



p=range(2)
d=range(1)
q=range(2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

min_aic_list=[]
for param in pdq:
	for param_seasonal in seasonal_pdq:
		try:
			mod = sm.tsa.statespace.SARIMAX(ts,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
			results = mod.fit()
			min_aic_list.append([results.aic,param,param_seasonal])
		except:
		   continue
	            
min_aic_df=pd.DataFrame(min_aic_list,columns=["aic","param","param_seasonal"])




mod = sm.tsa.statespace.SARIMAX(ts,
                                order=(1, 2, 4),
                                seasonal_order=(0, 2, 2, 3),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


results.plot_diagnostics(figsize=(15, 12))
plt.show()


pred=results.get_prediction(start=pd.to_datetime('2010-01-01'),dynamic=False)
print("MSE:{}".format(mean_squared_error(ts.ix["2010-01":],pred.predicted_mean)))
print("RMSE:{}".format(np.sqrt(mean_squared_error(ts.ix["2010-01":],pred.predicted_mean))))
print("r2 score:{}".format(r2_score(ts.ix["2010-01":],pred.predicted_mean)))



pred_f = results.get_forecast(steps=17)
pred_f.predicted_mean

plt.plot(pd.concat([ts,pred_f.predicted_mean]))


data_inf=[]
data_inf.append(list(data_pre.columns))



'''exog=data_pre.drop(["Year","Month of absence","Absenteeism time in hours"],axis=1).values
ex_array=exog.values
ex=np.asarray()

ex = np.empty([37, 50])

mod = sm.tsa.statespace.SARIMAX(ts,order=(0, 2, 2),
                                seasonal_order=(0, 2, 2, 3),
                                enforce_stationarity=False,
                                enforce_invertibility=False,exog=[[37,50]]).fit(ex_array)
results = mod.fit()



arima = ARIMA(ts, exog=exog, order=(2,2,0),freq='B')
results = arima.fit()'''




