import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler  #data processing
from sklearn.tree import DecisionTreeClassifier   #Evaluating the model and training the Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#---read Datasets---
df=pd.read_csv("loan_borowwer_.csv")
df      #--- To display the datasets ---
#---Datasets are transformed for data analysis
del df["porpose"]
df.head()
df.isnull().sum()
x=df.drop("not.fully.paid",axis=1)
y=df["not.fully.paid"]
xtrain,xtest,ytrain,yteat=train_test_split(x,y,test_size=0.2,random_state=4)
s=StandardScaler()
xtrain=s.fit_transform(xtrain)
xtest=s.transform(xtest)
d=DecisionTreeClassifier (random_state=4)
d.fit(xtrain,ytrain)
ypred=d.predict(xtest)
print(accuracy_score(ytest,ypred))
a=np.array
a=[0,0.1392,853.43,11.264464,16.48,732,4740.0000,37879,69,6,0,0)
d.predict([a])
confusion_matrix(ytest,ypred)
