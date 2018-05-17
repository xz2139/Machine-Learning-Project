import pandas as pd
import pickle
import numpy as np

from sklearn import linear_model


reduced2=pd.read_csv('data_bio_sumed_pred.csv')

def appeal(x):
    if x=='Rev':
        return 0
    elif x=='Aff':
        return 1
    else:
        return None
    
reduced2['Prediction']=reduced2.pred.apply(appeal)

reduced2['Intercept']=1
##Prediction
print('Prediction')
print(reduced2.dropna(subset = ['Prediction']).groupby('Prediction').length_dif_dm.describe())
##Real Target
print('Real Target')
print(reduced2.dropna(subset = ['Prediction']).groupby('Affirmed').length_dif_dm.describe())

def cat(v):
    if v>(-0.5) and v<0.5:
        return 'NoChange'
    elif v>=0.5:
        return 'Increase'
    else:
        return 'Decrease'

reduced2['CatLength']=reduced2.length_3m_dif.apply(cat)

print(reduced2.dropna(subset = ['Prediction']).groupby('Prediction').CatLength.value_counts())
print(reduced2.dropna(subset = ['Prediction']).groupby('Affirmed').CatLength.value_counts())

regr1 = linear_model.LinearRegression()
regr1.fit(reduced2.dropna(subset = ['Prediction'])[['Affirmed','Intercept']],reduced2.dropna(subset = ['Prediction']).length_dif_dm)
print(regr1.coef_[0])
regr2 = linear_model.LinearRegression()
regr2.fit(reduced2.dropna(subset = ['Prediction'])[['Prediction','Intercept']],reduced2.dropna(subset = ['Prediction']).length_dif_dm)
print(regr2.coef_[0])
