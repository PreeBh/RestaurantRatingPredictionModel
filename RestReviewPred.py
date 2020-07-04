# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
# using Random Forest Classifier now

os.chdir("C:\\Users\\bhatt\\.spyder-py3")
print (os.getcwd())
rest=pd.read_csv("TA_restaurants_curated.csv")
print(rest.head())
rest1=rest.drop(['URL_TA','ID_TA','Unnamed: 0'],axis=1)
numerical_nan=[feature for feature in rest1.columns if rest1[feature].isnull().sum()>1 and rest1[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values
for feature in numerical_nan:
    print("{}: {}% missing value".format(feature,np.around(rest1[feature].isnull().mean(),2)))
    numerical_nan1=[feature for feature in rest1.columns if rest1[feature].isnull().sum()>1 and rest1[feature].dtypes=='O']

## We will print the numerical nan variables and percentage of missing values
for feature in numerical_nan1:
    print("{}: {}% missing value".format(feature,np.around(rest1[feature].isnull().mean(),2)))
    
    rest1['Ranking'] = rest1['Ranking'].fillna(rest1['Ranking'].median())
rest1['Rating'] = rest1['Rating'].fillna(rest1['Rating'].median())
rest1['Number of Reviews'] = rest1['Number of Reviews'].fillna(rest1['Number of Reviews'].median())

from sklearn.preprocessing import LabelEncoder
rest1['Cuisine Style']= rest1['Cuisine Style'].astype(str)
lb_make = LabelEncoder()
rest1['Cuisine Style1'] = lb_make.fit_transform(rest1['Cuisine Style'])
#rest1[['Cuisine Style', 'Cuisine Style1']].head(11)
rest1['Price Range']= rest1['Price Range'].astype(str)

lb_make = LabelEncoder()
rest1['Price Range1'] = lb_make.fit_transform(rest1['Price Range'])
#rest1[['Price Range', 'Price Range1']].head(11)

rest1=rest1.drop(['Cuisine Style','Price Range'],axis=1)
rest1=rest1.drop(['Name','City', 'Reviews'],axis=1)

#changing the data of dependent feature 

rest1['Rating'].unique()
rest1['Rating']= rest1['Rating'].astype(int)
rest1.info()

#changing into dependent and independent feature 
x = rest1.iloc[:,[0,2,3,4]]
y = rest1['Rating']

#spilitting the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.30,random_state=105)
print(x_train.head())


clf1=RandomForestClassifier(n_estimators=100)
clf1.fit(x_train,y_train)
y_pred1=clf1.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))

# Saving model to disk
pickle.dump(clf1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))