import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
#pre_processing
dataset=pd.read_csv('50_Startups.csv')  
X=dataset.iloc[:,:-1].values          
Y=dataset.iloc[:,4].values

#for city variable implementing dummy variable using hot encoder
#3rd indexed col is  string col so change into labelencoderandhotencoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#to avoid dummy variable trap
X=X[:,1:] 


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

'''#feature Scaling
from sklearn.preprocessing import StandardScaler #no use on y
sc_X=StandardScaler()  # making object
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)         #test is part  so only fit '''


#multiple linear regression implementation
#same as linear regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train) #implementing regression


#for test data
Y_pred=regressor.predict(X_test)


#bulding model of back elimination
#adding X0 values(ones vector ) into matrix
import statsmodels.formula.api as sm
#X=np.append(arr=X,val=np.ones((50,1)).astype(int),axis=1)
#this adds ones to the last column
X=np.append(np.ones((50,1)).astype(int),X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()              #ordinary least square model
  #endog contains dependent variable
#exdog contains X_opt                                #OLS is method of class sm    
  
regressor_ols.summary()        #info about p value
#removing column of high p >0.05
#here x2 i.e col 3 index 2 has highest p so removing that col 
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit() 
regressor_ols.summary()  
  #for removal seee X values and X_opt
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit() 
regressor_ols.summary() 

X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit() 
regressor_ols.summary() 

X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit() 
regressor_ols.summary() 


