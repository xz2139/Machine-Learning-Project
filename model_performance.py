'''
experimented with different algorithm to see the predictive power of the text features
'''
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 


#Data preparation:

data = pd.read_csv("cc_merged_0429.csv",sep='\t',engine='python')
data.head(5)


data.info()
data.columns.values

data_use = data[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','txt','length_3m_dif']]

data_use.head(5)
data_use['txt'].iloc[0]



#train test split:
from sklearn.cross_validation import train_test_split
from numpy.random import RandomState

RS1 = RandomState(1)
train_data = data_use.sample(frac = 0.75, random_state = 200)
test_data = data_use.drop(train_data.index)

x_train = train_data['txt']
x_test = test_data['txt']
Y_train = train_data['length_3m_dif']
Y_test = test_data['length_3m_dif']

print(train_data.shape)
print(test_data.shape)



# tokenize the text using tfidf and count vectorizer, select the vectorizer
# that perform better for our task


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfid_vectorizer_2gram = TfidfVectorizer(stop_words = 'english', ngram_range= (1,2), binary = False)
tfid_vectorizer_2gram.fit(x_train)

#tfid_vectorizer
tf_train_2gram = tfid_vectorizer_2gram.transform(x_train)
tf_test_2gram = tfid_vectorizer_2gram.transform(x_test)


from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#experiment with different algorithm to get an understanding of the predictive power of the text features.

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


#random forest regressor

rfr = RandomForestRegressor()
rfr.fit(tf_train_2gram, Y_train)

rfr_base_tfidf_2gram_pred = rfr.predict(tf_test_2gram)
rfr_tfidf_2gram_base_mse = mean_squared_error(Y_test, rfr_base_tfidf_2gram_pred)
rfr_tfidf_2gram_base_mae = mean_absolute_error(Y_test, rfr_base_tfidf_2gram_pred)

print('rfr mean squared error is {}'.format(rfr_tfidf_2gram_base_mse))
print('rfr mean absolute error is {}'.format(rfr_tfidf_2gram_base_mae))


#decision tree regressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(tf_train_2gram, Y_train)

dtr_base_tfidf_2gram_pred = dtr.predict(tf_test_2gram)
dtr_tfidf_2gram_base_mse = mean_squared_error(Y_test, dtr_base_tfidf_2gram_pred)
dtr_tfidf_2gram_base_mae = mean_absolute_error(Y_test, dtr_base_tfidf_2gram_pred)

print('dtr mean squared error is {}'.format(dtr_tfidf_2gram_base_mse))
print('dtr mean absolute error is {}'.format(dtr_tfidf_2gram_base_mae))



#support vector regressor

svrr = SVR(C=1.0, epsilon=0.2)
svrr.fit(tf_train_2gram, Y_train)

svrr_base_tfidf_2gram_pred = svrr.predict(tf_test_2gram)
svrr_tfidf_2gram_base_mse = mean_squared_error(Y_test, svrr_base_tfidf_2gram_pred)
svrr_tfidf_2gram_base_mae = mean_absolute_error(Y_test, svrr_base_tfidf_2gram_pred)

print('svrr mean squared error is {}'.format(svrr_tfidf_2gram_base_mse))
print('svrr mean absolute error is {}'.format(svrr_tfidf_2gram_base_mae))


#gradient boosting accept sparse matrix as input in "fit" method, but currently does not accept sparse matrix for "predict" method
#gradient boosting regressor:

gbr = GradientBoostingRegressor()
gbr.fit(tf_train_2gram, Y_train)

gbr_base_tfidf_2gram_pred = gbr.predict(tf_test_2gram.todense())
gbr_tfidf_2gram_base_mse = mean_squared_error(Y_test, gbr_base_tfidf_2gram_pred)
gbr_tfidf_2gram_base_mae = mean_absolute_error(Y_test, gbr_base_tfidf_2gram_pred)

print('gbr mean squared error is {}'.format(gbr_tfidf_2gram_base_mse))
print('gbr mean absolute error is {}'.format(gbr_tfidf_2gram_base_mae))


'''from the experiment, we see that gradient boosting tree has the best performance, however, overall, the text feature doesn't predicte sentencing length very well according to mean absolute error and mean squared error.'''

#plt.scatter(Y_test,gbr_base_tfidf_2gram_pred)
#plt.show()
#savefig('GBR_predicted_VS_true.png')




