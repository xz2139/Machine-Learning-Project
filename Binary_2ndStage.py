
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

##Read in predicted Variable
preds=pd.read_csv('data_bio_sumed_pred.csv')


# In[12]:


##Convert category into float
def predict(x):
    if x=='Rev':
        return 0
    elif x=='Aff':
        return 1
    else:
        return None

def original(x):
    if x==2:
        return 0
    elif x==1:
        return 1
    else:
        return None
preds['Prediction']=preds.pred.apply(predict)
preds['Real']=preds.res.apply(original)


##Add intercept
preds['Intercept']=1


# In[13]:


preds


# In[14]:



##Prediction statistic describtion

print('Prediction')
print(preds.dropna(subset = ['Prediction']).groupby('pred').length_3m_dif.describe())

##Real Target statistic describtion
## 1 means affirmed, 0 meansa reversed
print('Real Target')
print(preds.dropna(subset = ['Prediction']).groupby('Real').length_3m_dif.describe())


# In[18]:



##If categorize, use here
def cat(v):
    if v>(-0.5) and v<0.5:
        return 'NoChange'
    elif v>=0.5:
        return 'Increase'
    else:
        return 'Decrease'

preds['CatLength']=preds.length_3m_dif.apply(cat)

print(preds.dropna(subset = ['Prediction']).groupby('Prediction').CatLength.value_counts())
print(preds.dropna(subset = ['Prediction']).groupby('Real').CatLength.value_counts())


# In[26]:


##Fit Linear regression and get coefficients
regr1 = linear_model.LinearRegression()
regr1.fit(preds.dropna(subset = ['Prediction'])[['Real','Intercept']],preds.dropna(subset = ['Prediction']).length_3m_dif)
print('Real Binary OLS Coefficient:',regr1.coef_[0])
regr2 = linear_model.LinearRegression()
regr2.fit(preds.dropna(subset = ['Prediction'])[['Prediction','Intercept']],preds.dropna(subset = ['Prediction']).length_3m_dif)
print('Prediction Binary OLS Coefficient:',regr2.coef_[0])


# In[39]:


from sklearn.metrics import mean_squared_error
print('MSE of Binary second Stage: ',mean_squared_error(preds.dropna(subset = ['Prediction'])['Real'],regr2.predict(preds.dropna(subset = ['Prediction'])[['Prediction','Intercept']])))


# In[29]:


import seaborn as sns
sns.set_style('darkgrid')


# In[33]:


plt.scatter(preds.dropna(subset = ['Prediction'])['Real'],preds.dropna(subset = ['Prediction']).length_3m_dif)
plt.show()


# In[34]:


plt.scatter(preds.dropna(subset = ['Prediction'])['Prediction'],preds.dropna(subset = ['Prediction']).length_3m_dif)
plt.show()


# In[32]:


plt.scatter(preds.dropna(subset = ['Prediction']).length_3m_dif, regr2.predict(preds.dropna(subset = ['Prediction'])[['Prediction','Intercept']]))
plt.xlabel('Real Sentencing length')
plt.ylabel('Predicted Sentencing Length')
plt.show()

