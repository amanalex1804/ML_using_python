import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#pre_processing
dataset=pd.read_csv('Position_Salaries.csv')  
X=dataset.iloc[:,1:2].values     #level itself decides     
Y=dataset.iloc[:,2].values



'''
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)'''

#feature Scaling
from sklearn.preprocessing import StandardScaler #no use on y
sc_X=StandardScaler()  # making object
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=Y.reshape(-1,1)
Y=sc_Y.fit_transform(Y)       #test is part  so only fit


#without feature scaling the regressor will yield a st. line and red of 13k 
#  while ans is 17k


#svr
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)


#visualising regressor over the data set
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.show()

#predicting the result
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))



