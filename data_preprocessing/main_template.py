import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
dataset=pd.read_csv('Data.csv')  #save the file where there is dataset
#index in python starts with 0 
X=dataset.iloc[:,:-1].values          #gives output of independent variables i.e
                                   # last column is dependent col so ignoring that
Y=dataset.iloc[:,3].values         # y is in 4th col but index is from 0    
#splitting dataset into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

'''#feature Scaling
from sklearn.preprocessing import StandardScaler #no use on y
sc_X=StandardScaler()  # making object
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)         #test is part  so only fit '''