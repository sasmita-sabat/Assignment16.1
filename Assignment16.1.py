
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
#load dataset
affairdata=sm.datasets.fair.load_pandas().data
#add "affair" column :1  represents having affairs,0 represents not
affairdata['affair']=(affairdata.affairs > 0).astype(int)
#create dataframes  with an intercept column  and dummy variables for  occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',dta, return_type="dataframe")
print(X.columns)
#Renaming the column names
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',

'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})
#convert y into 1-d array
y = np.ravel(y)
#Logistic regression model
model=LogisticRegression()
model=model.fit(X,y)
#check the accurancy of the training set
model.score(X,y)
# means 73% accuracy
y.mean()
#32%  of the women had affairs.

#To determine the coefficients
pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))))

#ModelEvaluation  using a validation set
#using a training set and test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)
model2=LogisticRegression()
model2.fit(X_train,y_train)
predicted=model2.predict(X_test)
probs=model2.predict_proba(X_test)
metrics.accuracy_score(y_test,predicted)
metrics.roc_auc_score(y_test,probs[:,1])
metrics.confusion_matrix(y_test,predicted)
metrics.classification_report(y_test,predicted)
#model evaluation using cross validation
scores=cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=10)
scores.mean()

#Predict using a data 
#Assume  a 32-year-old house wife who graduated college, has been married for 5 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

X=np.array([[1,0,0,1,0,0,1,0,0,0,0,4,32,5,1,3,16]])
model.predict_proba(X)
#probability of the affair is 15%

