import pandas as pd
import numpy as np
data=pd.read_excel("C:/Users/mudmoham/Documents/pr/case studies/Employee Absenteeism/Absenteeism_at_work_Project.xls",sheetname="Year_Sheet")

pd.isnull(data).sum()
data["Reason for absence"]=data["Reason for absence"].fillna(20)
data.shape
data=data[data["Reason for absence"]!=0]

from fancyimpute import KNN
data=pd.DataFrame(KNN(k=3).complete(data),columns=data.columns)
data=data.apply(np.round,axis=1)


pd.isnull(data).sum()
for col in num_columns:
	q75,q25=np.percentile(data[col],[75,25])
	iqr=q75-q25
	maximum=q75+iqr*1.5
	minimum=q25-iqr*1.5
	data.loc[data[col]<minimum,col]=np.nan
	data.loc[data[col]>maximum,col]=np.nan

data=pd.DataFrame(KNN(k=3).complete(data),columns=data.columns)
data=data.apply(np.round,axis=1)

data=data.drop(["Weight","Height","Disciplinary failure"],axis=1)



data["BMI"]=pd.cut(data["Body mass index"],[0,18.5,24.9,29.9,40],labels=["underweight","normal weight","overweight","obesity"])
data["Age"]=pd.cut(data["Age"],[17,34,55,80],labels=["Young Adults","Middle Aged","Old"])
data["Education"]=data["Education"].replace({1:"High School",2:"Graduate",3:"Post Graduate",4:"Master and Doctor"})
data["Son"]=pd.cut(data["Son"],[0.0,0.99,2.99,4],labels=["No Children","Less Child Count","More Child Count"])
data["Son"]=data["Son"].fillna("No Children")
data["Pet"]=pd.cut(data["Pet"],[0.0,0.99,2.99,8],labels=["No Pets","Less Pets","More Pets"])
data["Pet"]=data["Pet"].fillna("No Pets")
data["Service time"]=data["Service time"].replace(29,19)
data["Service time"]=pd.cut(data["Service time"],[1.0,3.99,9.99,15.99,21.99,24],labels=["Mid-Night","Morning","Noon","Evening","Night"])
data["Service time"]=data["Service time"].fillna("Mid-Night")
data["Day of the week"]=data["Day of the week"].replace({2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday"})
#data=data.drop(["Seasons"],axis=1)
#data["Season"]=pd.cut(data["Month of absence"],[0,3.99,6.99,8.99,12],labels=["Summer","Autumn","Winter","Spring"])
data["Seasons"]=data["Seasons"].replace({1:"Summer",2:"Autumn",3:"Winter",4:"Spring"})
di={1:"infectious diseases",2:"Cancer Related",3:"Immune Related",4:"Nutrional Related",5:"Mental Disorders",6:"Mental Disorders",7:"Eye and Ear",8:"Eye and Ear",9:"Heart Related",10:"Respiratory",11:"Digestive",12:"Skin",13:"Physical Issues",14:"genitourinary",15:"Maternity,Paternity",16:"Abnormal Diseases",17:"Abnormal Diseases",18:"Abnormal Diseases",19:"Injury,Poisoning",20:"External Factors",21:"External Factors",22:"Follow Up",23:"Consultations",27:"Consultations",28:"Consultations",24:"Blood Donations",25:"Lab",26:"Unjustified Absence"}
data["Reason for absence"]=data["Reason for absence"].replace(di)



data=data.loc[:,['Year','Month of absence','Reason for absence','Age','Work load Average/day ','Absenteeism time in hours']]
data_encode=pd.get_dummies(data.iloc[:,[2,3]],drop_first=True)
data_encode1=pd.concat([data,data_encode],axis=1)
data_encode1=data_encode1.drop(['Reason for absence','Age'],axis=1)

data_pre=pd.DataFrame(data_encode1.groupby(["Year","Month of absence"],as_index=False).mean())
#data_pre1=pd.DataFrame(data_encode1.groupby(["Year","Month of absence"],as_index=False)["Absenteeism time in hours"].sum())

for i in range(1,12):
	data_pre["T_"+str(i)]=data_pre["Absenteeism time in hours"].shift(i)

#data_pre=data_pre.drop(["Absenteeism time in hours"],axis=1)
#data_pre["Absenteeism time in hours"]=data_pre1["Absenteeism time in hours"]


	
data_pre=data_pre.fillna(0)
train=data_pre[round(data_pre["Year"])!=2010]
test=data_pre[round(data_pre["Year"])==2010]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train_data=pd.DataFrame(scaler.fit_transform(train),columns=train.columns)
train_data_X=train_data.drop(["Absenteeism time in hours"],axis=1)

test_data=pd.DataFrame(scaler.fit_transform(test),columns=test.columns)
test_data_X=test_data.drop(["Absenteeism time in hours"],axis=1)


from sklearn.ensemble import RandomForestRegressor
dt=RandomForestRegressor()
dt.fit(train_data_X,train_data["Absenteeism time in hours"])
y_pred=pd.DataFrame(dt.predict(test_data_X))
y_pred_i=pd.DataFrame(scaler.inverse_transform(pd.concat([test_data_X,y_pred],axis=1)),columns=test_data.columns).iloc[:,29]

from sklearn.metrics import mean_squared_error
print(mean_squared_error(test["Absenteeism time in hours"],y_pred_i))




xtrain_reshape=train_data_X.values.reshape((train_data_X.shape[0],1,train_data_X.shape[1]))
xtest_reshape=test_data_X.values.reshape((test_data_X.shape[0],1,test_data_X.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model_k=Sequential()
model_k.add(LSTM(1,input_shape=(xtrain_reshape.shape[1], xtrain_reshape.shape[2])))
model_k.add(Dense(1))
model_k.compile(loss='mae', optimizer='adam')

history = model_k.fit(xtrain_reshape, train_data["Absenteeism time in hours"], epochs=50, batch_size=72, validation_data=(xtest_reshape, test["Absenteeism time in hours"]), verbose=2, shuffle=False)
y_pred=model_k.predict(xtest_reshape)
y_pred=pd.DataFrame(y_pred)
y_pred_i=pd.DataFrame(scaler.inverse_transform(pd.concat([test_data_X,y_pred],axis=1)),columns=test_data.columns).iloc[:,29]
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test["Absenteeism time in hours"],y_pred_i))


