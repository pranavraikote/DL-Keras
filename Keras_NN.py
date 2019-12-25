# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:14:21 2019

@author: Pranav
"""

#Import Modules
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#Read data from the dataset
data = pd.read_csv('data.csv')

#Split data into parameters and target class
X = data.iloc[:,0:8]
Y = data.iloc[:,-1]

#Split the data into training and testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

#Define our neural network model (Sequential)
model = Sequential()

#This line will do two functions, define the input layer with 
#no. of neurons defined in input_dim and add the first Dense Layer
#Adding new layers is easy using model.add and defining the layer
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
#model.add(4, activation ='softmax')

#Compile the model with appropriate Optimizer and Loss functions
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Train the model with no. of epochs for a given batch size
model.fit(X_train, Y_train, epochs = 150, batch_size = 10)

#Calculate accuracy metrics
loss, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy is %.2f'%(accuracy*100))