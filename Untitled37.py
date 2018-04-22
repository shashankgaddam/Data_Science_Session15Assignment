
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
print(boston.keys())
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
from sklearn.linear_model import LinearRegression
z = boston.feature_names
x = bos[z]
y = bos.PRICE
lm = LinearRegression()
lm.fit(x,y)
a = lm.predict(x)
print(a)

