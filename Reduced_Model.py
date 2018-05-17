import pandas as pd
import pickle
from collections import Counter
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import ensemble
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

rf=pd.read_csv('bio_txt.csv')
cols=[col for col in rf if col.startswith('x_')]
X=rf[cols]
Y=rf.length_3m_dif

n_X=(X-X.min())/(X.max()-X.min())
n_y=(Y-Y.min())/(Y.max()-Y.min())

X_train, X_test, y_train, y_test = train_test_split(n_X, n_y, test_size=0.25, random_state=42)

X_train=np.hstack((X_train,np.ones((X_train.shape[0],1))))
X_test=np.hstack((X_test,np.ones((X_test.shape[0],1))))

### This is the best performing model after lots of trying and validation
X = np.nan_to_num(X_train)
y = y_train.values
regr = RandomForestRegressor(max_depth=11, random_state=44)
regr.fit(X, y)

print('R^2 of best reduced form regressor model: ',r2_score(y_test.values, regr.predict(np.nan_to_num(X_test))))
##0.106690850879

print('MSE of best reduced form regressor model: ',mean_squared_error(y_test.values, regr.predict(np.nan_to_num(X_test))))
##0.00185661687535

d=(sorted(zip(map(lambda x: round(x, 10), regr.feature_importances_), list(rf[cols].columns)+['intercept']), 
             reverse=True))
print('feature_importances:')
importances=[]
features=[]
for (i,j) in d[:10]:
    importances.append(i)
    features.append(j)
    print(j,i)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(d[:10])), importances[::-1], color='b', align='center')
plt.yticks(range(len(d[:10])), features[::-1])
plt.xlabel('Relative Importance')
plt.show()

plt.scatter(y_test.values,regr.predict(np.nan_to_num(X_test)))
plt.xlabel('Real_target')
plt.ylabel('Predicted_target')
plt.show()

########this is when turning this into classification problem by categorize sentencing length changes

def cat(v):
    if v>(-0.5) and v<0.5:
        return 0
    elif v>=0.5:
        return 1
    else:
        return -1
    
X=rf[cols]
Y=rf.length_3m_dif

n_X=(X-X.min())/(X.max()-X.min())

X_train, X_test, y_train, y_test = train_test_split(n_X, Y, test_size=0.25, random_state=42)   

classifier =ensemble.GradientBoostingClassifier(max_depth=5)
classifier.fit(X,y)


y_pred = classifier.predict(np.nan_to_num(X_test))
reversefactor = dict(zip([-1,0,1],['Decrease','NotChanged','Increase']))
y_testing = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_testing, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(f1_score(y_testing, y_pred, labels=None, average='micro', sample_weight=None))
