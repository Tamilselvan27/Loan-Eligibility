import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
#---read Datasets---
df=pd.read_csv("loan_borowwer_.csv")
df      #--- To display the datasets ---
#---Datasets are transformed for data analysis
del df["porpose"]
df.head()
df.isnull().sum()

