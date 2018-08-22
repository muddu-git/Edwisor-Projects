'''Importing required libraries''' 
import os
import pandas as pd
import numpy as np
#from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

'''Reading files'''
os.chdir('C:/Users/mudmoham/Documents/pr/Churn Reduction/Input')
train=pd.read_csv("Train_data.csv")
test=pd.read_csv("Test_data.csv")
data=pd.concat([train,test]).reset_index(drop=True)

'''Outlier Analysis-Replacing the outliers with 0.95 and 0.05 percentile values'''
Num_Columns=[]
Cat_Columns=[]
for col in data.columns:
	if data[col].dtype==np.int64 or data[col].dtype==np.float64:
		Num_Columns.append(col)
	else:
		Cat_Columns.append(col)
		
con_data=data[Num_Columns]
cat_data=data[Cat_Columns]

for col in con_data.columns:
	q75,q25=np.percentile(data[col],[75,25])
	iqr=q75-q25
	minimum=round(q25-(iqr*1.5))
	maximum=round(q75+(iqr*1.5))
	data.loc[data[col]<minimum,col]=minimum
	data.loc[data[col]>maximum,col]=maximum
	
#Converting Object Variables to categorical codes
for col in data.columns:
	if data[col].dtype==np.object:
		data[col]=pd.Categorical(data[col])
		data[col]=data[col].cat.codes

'''Feature Selection-Chi Square test of independence for finding redundancy in categorical variables'''
#print('''*********************Chi-Square Test*********************''')
#for col in Cat_Columns:
#	if col!="Churn":
#		chi2,p,dof,exp=chi2_contingency(pd.crosstab(data["Churn"],data[col]))
#		print(col)
#		print("p value is {} degree of freedom is {} test statistic is {}".format(p,dof,chi2))
#print('''*********************************************************''')

#High Correlation:total day charge,total day minutes,total eve charge,total eve minutes,total night minutes,total night charge,total intl charge,total intl minutes
drop_columns=["phone number","total day minutes","total eve minutes","total night minutes","total intl minutes"]
data=data.drop(drop_columns,axis=1)

'''Feature Engineering'''
data["total calls"]=data["total day calls"]+data["total eve calls"]+data["total night calls"]+data["total intl calls"]
data["total call charge"]=data["total day charge"]+data["total eve charge"]+data["total night charge"]+data["total intl charge"]
data=data.drop(["total day calls","total eve calls","total night calls","total intl calls","total day charge","total eve charge","total night charge","total intl charge"],axis=1)

'''Feature Scaling-Normalization'''
data=pd.DataFrame(MinMaxScaler().fit_transform(data),columns=data.columns)

#Sampling-dividing data into training and test data
train_data=data.iloc[0:train.shape[0],:]
test_data=data.iloc[train.shape[0]:,:].reset_index(drop=True)
x_train,y_train=train_data.drop("Churn",axis=1),train_data["Churn"]
x_test,y_test=test_data.drop("Churn",axis=1),test_data["Churn"]

'''******************************** Model Selection-Random Forest ***********************************************'''
rf=RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='entropy', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0,
            n_estimators=300, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
rf.fit(x_train,y_train)

#predicting using optimal probability threshold:0.3
y_output=(rf.predict_proba(x_test)[:,1]>0.3).astype(int)

print('''************ Model Metrics with Threshold 0.3 ****************''')
cm=pd.crosstab(y_test,y_output)
TN=cm.iloc[0,0]
FP=cm.iloc[0,1]
FN=cm.iloc[1,0]
TP=cm.iloc[1,1]
print("Accuracy is {}".format(accuracy_score(y_test,y_output)*100))
#negative cases prediction
print("Specificity or True Negative Rate is {}".format(TN*100/(TN+FP)))	
#postive cases prediction
print("Recall or sensitivity or True Postive Rate is {}".format(TP*100/(TP+FN)))
print("False Positive Rate is {}".format(FP*100/(FP+TN)))
print("False Negative Rate is {}".format(FN*100/(FN+TP)))
print("************************************************************")

print('''*********  OUTPUT *************''')
print("Churn Score={:5.2f}%".format(np.mean(y_output)*100))
print("Number of persons who are churning out:{}".format(np.sum(y_output==1)))
print("Number of persons who are not churning out:{}".format(np.sum(y_output!=1)))
print('''*******************************''')

#Writing output to file
output=pd.DataFrame(test)
y_output=y_output.astype(bool)
output["Churn Predicted"]=y_output
output["Churn Probabilities"]=rf.predict_proba(x_test)[:,1]
output.to_csv("C:/Users/mudmoham/Documents/pr/Churn Reduction/output/py_output.csv",index=False)










     
       
	
		   
        

 













		
	
	


	











	


	

