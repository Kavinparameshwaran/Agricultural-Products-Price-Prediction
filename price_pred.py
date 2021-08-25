import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset=pd.read_csv('price_list.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

labelEncoder=LabelEncoder()
x[:,0]=labelEncoder.fit_transform(x[:,0])
x[:,1]=labelEncoder.fit_transform(x[:,1])
x[:,2]=labelEncoder.fit_transform(x[:,2])
x[:,3]=labelEncoder.fit_transform(x[:,3])

#onehotEncoder=OneHotEncoder(categorical_features=[3])
#x=onehotEncoder.fit_transform(x).toarray()

x1=OneHotEncoder().fit_transform(x).toarray()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
# a=0
# b=0
# c=5
# y_pred=regressor.predict([[0,a,b,c]])
# print(y_pred,0)
# y_pred=regressor.predict([[1,a,b,c]])
# print(y_pred,1)
# y_pred=regressor.predict([[2,a,b,c]])
# print(y_pred,2)
# y_pred=regressor.predict([[3,a,b,c]])
# print(y_pred,3)
# y_pred=regressor.predict([[4,a,b,c]])
# print(y_pred,4)
# y_pred=regressor.predict([[5,a,b,c]])
# print(y_pred,5)


pickle.dump(regressor,open('model.pkl','wb'))
#model=pickle.load(open('model.pkl','wb'))
