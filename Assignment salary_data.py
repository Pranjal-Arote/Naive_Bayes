# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:05:57 2024

@author: Pranjal Arote

Problem Statement:
1.) Prepare a classification model using the Naive Bayes algorithm for the salary
dataset. Train and test datasets are given separately. Use both for model 
building.   
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
 
#####Loading data
salary_train=pd.read_csv("C:/DataSet/SalaryData_Train.csv",encoding="ISO-8859-1")
salary_test=pd.read_csv("C:/DataSet/SalaryData_Test.csv" ,encoding="ISO-8859-1")
 
from sklearn.model_selection import train_test_split
train_test_split(salary_train,test_size=0.2)
 
########creating matrix of token counts for entire text documents####

salary_bow=CountVectorizer().fit(salary_test.Salary) 
all_salary_matrix=salary_bow.transform(salary_test.Salary)

####For training clients 

train_salary_matrix=salary_bow.transform(salary_train.education)

###for testing clients
test_salary_matrix=salary_bow.transform(salary_test.education)

#####Learning Term weightaging and normaling on entire clients salary
tfidf_transformer=TfidfTransformer().fit(all_salary_matrix)

######preparing TFIDF
train_tfidf=tfidf_transformer.transform(train_salary_matrix)

####preparing TFIDF for test data
test_tfidf=tfidf_transformer.transform(test_salary_matrix)
test_tfidf.shape

######Now let us apply this to the Naive Bayes therorem

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,salary_train.workclass)

######Evalution on test data

test_pred_m= classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==salary_test.workclass) 
accuracy_test_m    
  




