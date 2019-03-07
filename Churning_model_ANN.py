# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:04:49 2018

@author: Adarsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# this one for first categorical variable that is Country
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])

#this one is for second categorical variable that is gender
labelencoder1=LabelEncoder()
X[:,2]=labelencoder1.fit_transform(X[:,2])


ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()

# To avoid dummy variable trap , we will exclude one column of the encoded part
X=X[:,1:]

# we dont have to encode gender as one hot encoder, becoz it has only 2 categorical variable and one will be eliminated becoz of dummy variable trap


#test train data split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



#feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)


# creating ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing ANN

#we create an object of Sequential class and that will be our ANN Model
classifier=Sequential()

# Adding input and first hidden layer
classifier.add(Dense(kernel_initializer='uniform',activation='relu',output_dim=6,input_dim=11))

# Adding second hidden layer

#here we dont need to pass input_dim, because it already knows the number of inputs from previous layer but the first layer wont knw this so we should pass input_dim saying create input nodes of size equal to input_dim
classifier.add(Dense(kernel_initializer='uniform',activation='relu',output_dim=6))

#output Layer

classifier.add(Dense(output_dim=1,kernel_initializer='uniform',activation='sigmoid'))

# Compiling ANN (i.e Adding stochastic gradient descent)

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# here optimizer is adam which is one of the stochastic gradient algorithm
# loss function here is cost function which is binary_crossentropy for classification
# if more than 2 classes we use loss=categorical_entropy
#metrics is accuracy that is how we measure the result for classification


# fit the model

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# predicting the output

y_pred=classifier.predict(X_test)

# we are changing y_pred from being probability to True or false for using it in Confusion matrix
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



