import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
dataset=pd.read_csv('Data.csv')  #save the file where there is dataset
#index in python starts with 0 
X=dataset.iloc[:,:-1].values          #gives output of independent variables i.e
                                   # last column is dependent col so ignoring that
Y=dataset.iloc[:,3].values         # y is in 4th col but index is from 0                                   