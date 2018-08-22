#Importing Libraries
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

#reading data
data=pd.read_excel("C:/Users/mudmoham/Documents/pr/case studies/Employee Absenteeism/Absenteeism_at_work_Project.xls",sheetname="Year_Sheet")

#Missing Value Analysis
data=data[data["Reason for absence"]!=0]

#KNN Imputation
data=pd.DataFrame(KNN(k=3).complete(data),columns=data.columns)
data=data.apply(np.round,axis=1)

#grouping by Months
data_t=pd.DataFrame(data.groupby(["Year","Month of absence"],as_index=False)["Absenteeism time in hours"].mean())
data_c=pd.DataFrame(data.groupby(["Year","Month of absence"],as_index=False)["ID"].count())
Two_years_data=data[(data["Year"]==2008) | (data["Year"]==2009)]
month_group=pd.DataFrame(Two_years_data.groupby(["Month of absence"],as_index=False)["ID"].mean())
month_group.columns=["Month","Average Number of Absentees per Month"]
data=pd.concat([data_t,data_c],axis=1).iloc[:,[0,1,2,5]]
data.columns=['Year', 'Month of absence',"Absenteeism time average","No of Absentees"]

#converting into time series

data["YearMonth"]=data["Year"].astype(int).astype(str)+"-"+data["Month of absence"].astype(int).astype(str)
data["YearMonth"]=pd.to_datetime(data["YearMonth"],format="%Y-%m")
data["YearMonth"]=data["YearMonth"].dt.strftime('%Y-%m')
data.index=data["YearMonth"]
ts=data["Absenteeism time average"]

#Dividing data into train and test 
train_ts=ts.ix["2007-07":"2009-12"]
test_ts=ts.ix["2010-01":]

#Modelling-Sarimax
mod = SARIMAX(ts,order=(0,1,1),seasonal_order=(0,1,1,4),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
pred=results.get_prediction(start=pd.to_datetime('2010-01-01'),dynamic=False)

#Evaluation
print("MSE:{}".format(mean_squared_error(ts.ix["2010-01":],pred.predicted_mean)))
print("RMSE:{}".format(np.sqrt(mean_squared_error(ts.ix["2010-01":],pred.predicted_mean))))

#Forecasting 2011 data
pred_f = results.get_forecast(steps=17)
output=pd.DataFrame(pred_f.predicted_mean["2011":],columns=["Average Absenteeism"]).reset_index(drop=True)

#Writing data to output
output=pd.concat([output,month_group],axis=1)
output["Total Absenteeism time in hours"]=output["Average Absenteeism"]*output["Average Number of Absentees per Month"]
output=output.drop(["Average Absenteeism","Average Number of Absentees per Month"],axis=1)
output.to_csv("C:/Users/mudmoham/Documents/pr/case studies/Employee Absenteeism/output/py_output_2011_absenteeism.csv",index=False)









