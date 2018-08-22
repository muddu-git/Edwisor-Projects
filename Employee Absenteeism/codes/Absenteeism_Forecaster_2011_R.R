#Loading required packages
x=c("readxl","VIM","zoo","fpp","forecast")
lapply(x, require, character.only = TRUE)

#Reading files
data=read_excel("C:/Users/mudmoham/Documents/pr/case studies/Employee Absenteeism/Absenteeism_at_work_Project.xls",sheet = 'Year_Sheet')
data$`Reason for absence`[is.na(data$`Reason for absence`)]=20
data=data[data$`Reason for absence`!=0,]
data$`Absenteeism time in hours`[data$`Absenteeism time in hours`==0]=1

#Imputing Missing Values
data=kNN(data, k=3)
data=subset(data,select=-c(23:44))


#Grouping by year and Month
data_a=aggregate(data,by=list(data$Year,data$`Month of absence`),FUN = mean)[c("Group.1","Group.2","Absenteeism time in hours")]
data_c=aggregate(data,by=list(data$Year,data$`Month of absence`),FUN = NROW)[c("Group.1","Group.2","Absenteeism time in hours")]["Absenteeism time in hours"]
Two_years_data=data[(data$Year ==2008) | (data$Year == 2009), ]
Two_years_data_g=aggregate(Two_years_data,by=list(Two_years_data$`Month of absence`),FUN=NROW)[,c(1,2)]
colnames(Two_years_data_g)=c("Month","No of Absentees")
data_g=cbind(data_a,data_c)
colnames(data_g)=c("Year","Month of Absence","Average Absenteeism","Absentees Count")
data_g=data_g[order(data_g$Year),]
data_g$year_month <- as.yearmon(paste(data_g$Year, data_g$`Month of Absence`,sep = "-"), "%Y-%m")
data_g$year_month=as.Date(data_g$year_month)
data_g=data_g[,c(3,4,5)]
data_g$year_month=format(data_g$year_month, format="%Y-%m")

#time series
tms=data_g$`Average Absenteeism`
tms=ts(tms,start=c(2007,7),end=c(2010,7),frequency=12)
#plot(tms)

#Training and Testing dataset
tms_train=window(tms,start=c(2007,7),end=c(2009,12))
tms_test=window(tms,start=c(2010,1),end=c(2010,7))
#plot(tms_train)
#plot(tms_test)

#SARIMA Model
model=arima(tms_train,order = c(0,1,1),seasonal = c(0,1,1,4))
#summary(model)
#confint(model)
#plot.ts(model$residuals)
#acf(model$residuals)

#Forecasting for 2009,2010 and 2011
model_forecast=forecast(model,h=24)
#model_forecast
#plot(model_forecast)


#Evaluation
accuracy(model_forecast,tms_test)

#Writing output
output=as.data.frame(model_forecast$upper)[13:24,1]
output_df=cbind(Two_years_data_g,output)
output_df$Total_absenteeism_time=output_df$`No of Absentees` * output_df$output
write.csv(output_df[,c(1,4)],file = "C:/Users/mudmoham/Documents/pr/case studies/Employee Absenteeism/output/R_output_2011_absenteeism.csv",row.names = F)





