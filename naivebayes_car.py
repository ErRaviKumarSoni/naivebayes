import numpy as np
import pandas as pd
#from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
cars = pd.read_csv('D:\\LIVE WIRE - DS\\Data Set\\cars.csv')
def norm_func(i):
    x = (i-i.min()) /(i.max()-i.min())
    return (x)
df_norm = norm_func(cars.iloc[:,0:3])
ip_columns = ["MPG","CYL","WGT","ENG"]
op_column = ["VS"]
type(cars)
# Splitting data into train and test
Xtrain,Xtest,ytrain,ytest = train_test_split(cars[ip_columns],cars[op_column],test_size=0.3, random_state=0)
#
"""model= GaussianNB()
model.fit(Xtrain,ytrain)
pred = model.predict(Xtest)
print(pred)
confusion_matrix(ytest,pred)
auc=(4+5)/(14)
auc#64%"""

