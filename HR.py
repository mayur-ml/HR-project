# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:15:26 2022
@author: mayur ingole
batch id:DSWDACOR 180722 B
""
# Q 4.	In the recruitment domain, HR faces the challenge of predicting if the
# candidate is faking their salary or not. For example, a candidate claims to have 5
# years of experience and earns 70,000 per month working as a regional manager. 
# he candidate expects more money than his previous CTC. 
# We need a way to verify their claims (is 70,000 a month working as a regional manager with
# an experience of 5 years a genuine claim or does he/she make less than that?) 
# Build a Decision Tree and Random Forest model with monthly income as the target variable. 
""
Business objective:new recruitment on best possible CTC 

predict = actual salary on each post with relevent years of employee experience
"""

#importing librarys
import pandas as pd # data manupulation
import numpy as np # numerical calculation
import matplotlib.pyplot as plt  # basic visulization
import seaborn as sns # advance visulization

#loding data set 
data=pd.read_csv("D:/DATA SCIENCE & ML/1 ASSIGNMENTS/data sets/Datasets_DT/HR_DT.csv")
      
                     ###########
                     ####EDA####
                     ###########
                     
data.info()
"""
Data columns (total 3 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   employee_Position   196 non-null    object  #CATEGORICALDATA
 1   Experience_iny      196 non-null    float64 #FLOT
 2    income_inM         196 non-null    int64   #CONTINIOUS DATA
dtypes: float64(1), int64(1), object(1)
"""
#NOT NULL VALUES

describe = data.describe()
"""
       Experience_iny     income_inM
count      196.000000     196.000000
mean         5.112245   74194.923469
std          2.783993   26731.578387
min          1.000000   37731.000000
25%          3.000000   56430.000000
50%          4.100000   63831.500000
75%          7.100000   98273.000000
max         10.500000  122391.000000
""
#work experience in years
avarage work experience people have in data set is 5 years 1 months
minimum work experience people have in data set is 1 year
maximum work experience people have in data set is 10 year

#income per months
avarage monthly income in this data set is 74194.923469
maximum  monthly income in this data set is 122391
minumum  monthly income in this data set is 37731
"""
#employee_Position
data['employee_Position '].value_counts()
sns.countplot(data['employee_Position ']) # visulization of position count
"""
Partner              28
Senior Partner       25
C-level              24
Region Manager       23
CEO                  23
Country Manager      18
Manager              17
Senior Consultant    16
Junior Consultant    14
Business Analyst      8
there are 10 unique employee position  
"""

#ploting histogram to see frequency of data distrubution 
#**INCOME IN MONTHS**
sns.histplot(data["Income_inM"],kde=True)
data.Income_inM.skew() #0.4533 skewness tels us direction of spread
data.Income_inM.kurt() #-1.171 kurtosis tels us extent of spread
"""
**INCOME IN MONTHS**
Data is right skewed maximum position have salary in 40000 to 80000
data is right skewed 0.4533 positive skewness
data has negative skewness data is platy kurtik
"""

#**workexperience is years**
sns.histplot(data["Experience_iny"],kde=True)
data.Experience_iny.skew()  #0.461 skewness tels us direction of spread
data.Experience_iny.kurt()  #-0.93 kurtosis tels us extent of spread
"""
maximum positions have experinece of 0 to 6 years 
data is right skewed 0.461 positive skewness
data has negative skewness data is platy kurtik -0.93
"""

                     ###########
                    #### DT #####
                     ###########

# creating dummy variable creation 0s and 1

position_label_maping = {'Business Analyst': 1, 'Junior Consultant': 2,'Senior Consultant': 3, 'Manager': 4, 'Country Manager': 5,'Region Manager': 6,'Partner': 7, 'Senior Partner': 8, 'C-level': 9, 'CEO': 10}
data['employee_Position '] = data['employee_Position '].map(position_label_maping)

#input and output split
predictors = data.iloc[:,data.columns!="Income_inM"]
target = data["Income_inM"]

#spliting data into test train using sklearn
#test size 25% of data train size 75% of data 
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(predictors,target,test_size=0.25,random_state=42 )

#imporing dicision tree regressdir from sklearn
from sklearn import tree
regtree = tree.DecisionTreeRegressor()
regtree.fit(x_train , y_train)

#PREDICTION
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

#measuring accuracy
from sklearn.metrics import mean_squared_error , r2_score
#error on test data
mean_squared_error(y_test,test_pred) 
#14299047.459183674

r2_score(y_test , test_pred)
# we get Rsquare of 0.98 on test dataset which is very good

#ERROR ON TRAIN DATASET
mean_squared_error(y_train , train_pred)
#720717.83

r2_score(y_train, train_pred)
# we get Rsquare of 0.999 which is almost 1 on train dataset



# let us try Random forest
from sklearn.ensemble import RandomForestRegressor
rf_clf = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)


# Error on test dataset
mean_squared_error(y_test, rf_clf.predict(x_test))
#12147018.518642858

r2_score(y_test, rf_clf.predict(x_test))
# we see that our mean squared error goes down
# we also see that our R squared is now 97.96% which is equal to 98% of DT
# we can use both random forest ad wells as DT

# let us now check if candidate is honest or fraud
# first we will have to add his entry to our x_test data
cand = {'employee_Position ': 6, 'Experience_iny': 5.0 }
x_test_cand = x_test.append(cand, ignore_index = True)
x_test_preds = rf_clf.predict(x_test_cand)
# since it was the last row that we added, we are only interested in predicted value of the last row


x_test_preds[-1]
# we get a value of 66703.054

"""
According to our model the candidate on position 6 
with 5 years of experience can have salary of 66703.054
if is taling HR department above 66703.054  he is taling lie 
"""











