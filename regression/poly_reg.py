import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#pre_processing
dataset=pd.read_csv('Position_Salaries.csv')  
X=dataset.iloc[:,1:2].values     #level itself decides     
Y=dataset.iloc[:,2].values



#no need of splitting as X contain only 1 col
'''from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler #no use on y
sc_X=StandardScaler()  # making object
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)         #test is part  so only fit '''

#poly linear regression

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X,Y)          #initial regression (learning)

#for polyfeatures
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(4)  #add degree here
X_poly=poly_reg.fit_transform(X)
#X_poly adds ones vector automatically
linear_reg_2=LinearRegression()
linear_reg_2.fit(X_poly,Y)    #learning final poly

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))

#visualising linear regression over the data set
plt.scatter(X,Y,color='red')
plt.plot(X,linear_reg.predict(X),color='blue')
plt.show()

#visualising poly_reg data
plt.scatter(X,Y,color='red')
plt.plot(X,linear_reg_2.predict(poly_reg.fit_transform(X)),color='blue')



#more visualisation
plt.scatter(X,Y,color='red')
plt.plot(X_grid,linear_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
