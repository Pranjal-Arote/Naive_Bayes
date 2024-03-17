# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:27:53 2024

@author: Pranjal Arote
In this case study, you have been given Twitter data collected from an anonymous
 twitter handle. With the help of a Naïve Bayes model, predict if a given tweet
about a real disaster is real or fake. 1 = real tweet and 0 = fake tweet

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
 
#####Loading data
disaster_data=pd.read_csv("C:/DataSet/Dister_Tweest_NB.csv",encoding="ISO-8859-1")

#########cleaning of data
import re
def cleaning_text(i):
    w=[]
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if(len(word)>3):
            w.append(word)
    return (" ".join(w))

##############Testing above functions with some test text
cleaning_text("Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all")
cleaning_text("#raining #flooding #Florida #TampaBay #Tampa 18 or 19 days. I've lost count ")
cleaning_text("Hii,How are you I am  Sad")

disaster_data.text=disaster_data.text.apply(cleaning_text)  
disaster_data=disaster_data.loc[disaster_data.text!="",:] 
 
from sklearn.model_selection import train_test_split
data_train,data_test=train_test_split(disaster_data,test_size=0.2)

########creating matrix of token counts for entire text documents####

def split_into_words(i):
    return[word for word in i.split(" ")]


data_bow=CountVectorizer(analyzer=split_into_words).fit(disaster_data.text)
all_data_matrix=data_bow.transform(disaster_data.text)

####For training messages

train_data_matrix=data_bow.transform(data_train.text)

###for testing messages
test_data_matrix=data_bow.transform(data_test.text)

#####Learning Term weightaging and normaling 
tfidf_transformer=TfidfTransformer().fit(all_data_matrix)

######preparing TFIDF 
train_tfidf=tfidf_transformer.transform(train_data_matrix)

####preparing TFIDF 
test_tfidf=tfidf_transformer.transform(test_data_matrix)
test_tfidf.shape

######Now let us apply this to the Naive Bayes therorem

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,data_train.target)

######Evalution on test data

test_pred_m= classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==data_test.target) 
accuracy_test_m    
  




 
