
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# # General Data Exploration

data = pd.read_csv("cc_merged_0429.csv",sep='\t',engine='python')
data.head(5)
data.info()
data.columns.values



#In this part, we are going to use Affirmed/Reversed decision of the circuit court, combined with the txt feature of each case to predict the difference in sentencing length


#Prepare data
data_use = data[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','txt','length_3m_dif']]




#Train test split:
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


# In[13]:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

count_vectorizer_1gram = CountVectorizer(stop_words = 'english', binary = False)
count_vectorizer_2gram = CountVectorizer(stop_words = 'english', ngram_range = (1,2) , binary = False)
tfid_vectorizer_1gram = TfidfVectorizer(stop_words = 'english', binary = False)
tfid_vectorizer_2gram = TfidfVectorizer(stop_words = 'english', ngram_range= (1,2), binary = False)

count_vectorizer_1gram.fit(x_train)
count_vectorizer_2gram.fit(x_train)

tfid_vectorizer_1gram.fit(x_train)
tfid_vectorizer_2gram.fit(x_train)


#count_vectorizer
count_train_1gram = count_vectorizer_1gram.transform(x_train)
count_test_1gram = count_vectorizer_1gram.transform(x_test)

count_train_2gram = count_vectorizer_2gram.transform(x_train)
count_test_2gram = count_vectorizer_2gram.transform(x_test)

#tfid_vectorizer
tf_train_1gram = tfid_vectorizer_1gram.transform(x_train)
tf_test_1gram = tfid_vectorizer_1gram.transform(x_test)

tf_train_2gram = tfid_vectorizer_2gram.transform(x_train)
tf_test_2gram = tfid_vectorizer_2gram.transform(x_test)

# In[9]:

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#count 1 gram:
lr_base_count_1gram = linear_model.LinearRegression()
lr_base_count_1gram.fit(count_train_1gram, Y_train)
lr_base_count_1gram_pred = lr_base_count_1gram.predict(count_test_1gram)

count_1gram_base_mae = mean_absolute_error(Y_test, lr_base_count_1gram_pred)
print('count_1gram_base_mae is ',count_1gram_base_mae)


count_1gram_base_mae_r2 = r2_score(Y_test, lr_base_count_1gram_pred)
print('count_1gram_base_mae_r2 is ',count_1gram_base_mae_r2)

plt.scatter(Y_test,lr_base_count_1gram_pred)
plt.xlabel('predicted difference in sentencing length')
plt.ylabel('actual difference in sentencing length')
plt.show()


#count 2 gram:
lr_base_count_2gram = linear_model.LinearRegression()
lr_base_count_2gram.fit(count_train_2gram, Y_train)
lr_base_count_2gram_pred = lr_base_count_2gram.predict(count_test_2gram)

count_2gram_base_mae = mean_absolute_error(Y_test, lr_base_count_2gram_pred)
print('count_2gram_base_mae is ',count_2gram_base_mae)

count_2gram_base_mae_r2 = r2_score(Y_test, lr_base_count_2gram_pred)
print('count_2gram_base_mae_r2 is ',count_2gram_base_mae_r2)

plt.scatter(Y_test,lr_base_count_2gram_pred)
plt.xlabel('predicted difference in sentencing length')
plt.ylabel('actual difference in sentencing length')
plt.show()


# tfidf 1 gram:
lr_base_tfidf_1gram = linear_model.LinearRegression()
lr_base_tfidf_1gram.fit(tf_train_1gram, Y_train)
lr_base_tfidf_1gram_pred = lr_base_tfidf_1gram.predict(tf_test_1gram)

tfidf_1gram_base_mae = mean_absolute_error(Y_test, lr_base_tfidf_1gram_pred)
print('tfidf_1gram_base_mae is ',tfidf_1gram_base_mae)

tfidf_1gram_base_mae_r2 = r2_score(Y_test, lr_base_tfidf_1gram_pred)
print('tfidf_1gram_base_mae_r2 is ',tfidf_1gram_base_mae_r2)

plt.scatter(Y_test,lr_base_tfidf_1gram_pred)
plt.xlabel('predicted difference in sentencing length')
plt.ylabel('actual difference in sentencing length')
plt.show()


# tfidf 2 gram:
lr_base_tfidf_2gram = linear_model.LinearRegression()
lr_base_tfidf_2gram.fit(tf_train_2gram, Y_train)
lr_base_tfidf_2gram_pred = lr_base_tfidf_2gram.predict(tf_test_2gram)

tfidf_2gram_base_mae = mean_absolute_error(Y_test, lr_base_tfidf_2gram_pred)
print('tfidf_2gram_base_mae is ',tfidf_2gram_base_mae)


tfidf_2gram_base_mae_r2 = r2_score(Y_test, lr_base_tfidf_2gram_pred)
print('tfidf_2gram_base_mae_r2 is ',tfidf_2gram_base_mae_r2)

plt.scatter(Y_test,lr_base_tfidf_2gram_pred)
plt.xlabel('predicted difference in sentencing length')
plt.ylabel('actual difference in sentencing length')
plt.show()


# In[ ]:



'''conclusion:
from the above experiment, we see that using the same number of grams, TfIdf vectorizer always performs better than count vectorizer according the Mean Absolute error and R square. For the same vectorizer, performance is better for larger range of grams. This intuitively make senese, however, there is a tradeoff between performance and computing resources needed. Based on the analysis, we choose to use TfIdf vectorizer with ngram_range =(1,2) for the following steps.'''


#experiment with different algorithm to get an understanding of the predictive power of the text features.
#provided in seperate .py file called model_performance.py



# fitting the text feature combined with the reversed/affirm decision into a neural network. To see how our data perform in a deep nerual net model.



from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

#input sparse matrix: tf_train_2gram, tf_test_2gram
svd = TruncatedSVD(n_components=80, n_iter=7, random_state=42)
svd.fit(tf_train_2gram)  
TruncatedSVD(algorithm='randomized', n_components=50, n_iter=7,
        random_state=42, tol=0.0)

svd_train = svd.transform(tf_train_2gram)
svd_test = svd.transform(tf_test_2gram)
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())

#################
##dataframe approach
svd_train_df = pd.DataFrame(data = svd_train, index=train_data.index)
svd_test_df = pd.DataFrame(data = svd_test, index = test_data.index)

svd_train_df[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','length_3m_dif']] = train_data[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','length_3m_dif']]
svd_test_df[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','length_3m_dif']] = test_data[['index','Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','length_3m_dif']]




# combined ['Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart'] into a binary variable
svd_train_df['res']=svd_train_df['Affirmed']+svd_train_df['AffirmedInPart']+svd_train_df['Reversed']+svd_train_df['ReversedInPart']+svd_train_df['Vacated']+svd_train_df['VacatedInPart']
svd_test_df['res']=svd_test_df['Affirmed']+svd_test_df['AffirmedInPart']+svd_test_df['Reversed']+svd_test_df['ReversedInPart']+svd_test_df['Vacated']+svd_test_df['VacatedInPart']


#keep only row with res ==1
svd_train_df=svd_train_df[svd_train_df['res']==1]
svd_test_df=svd_test_df[svd_test_df['res']==1]


def combine_reverse(row):
    if row['Reversed']==1:
        return 2
    
    elif row['Vacated']==1:
        return 2
    
    elif row['Affirmed']==1:
        return 1


svd_train_df['Res_binary'] = svd_train_df.apply(combine_reverse, axis=1)
svd_test_df['Res_binary'] = svd_test_df.apply(combine_reverse, axis=1)

svd_train_df=svd_train_df[(svd_train_df['Res_binary']==1) | (svd_train_df['Res_binary']==2)]
svd_test_df=svd_test_df[(svd_test_df['Res_binary']==1) | (svd_test_df['Res_binary']==2)]

svd_train_df= svd_train_df.drop(['Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','res'],axis=1)
svd_test_df= svd_test_df.drop(['Affirmed', 'AffirmedInPart', 'Reversed', 'ReversedInPart', 'Vacated', 'VacatedInPart','res'],axis=1)


svd_train_nn_df = svd_train_df.iloc[:,:]
svd_test_nn_df = svd_test_df.iloc[:,:]

svd_train_nn_df['y'] = svd_train_df['length_3m_dif']
svd_test_nn_df['y'] = svd_test_df['length_3m_dif']


svd_train_nn_df.drop(['length_3m_dif','index'],axis=1, inplace=True)
svd_test_nn_df.drop(['length_3m_dif','index'],axis=1,inplace=True)



#save the data (after dimension reduction)
svd_train_nn_df.to_csv("train_nn.csv", sep=',')
svd_test_nn_df.to_csv("test_nn.csv", sep=',')

#svd_train_nn_df = pd.read_csv("nn_prepared_svd_train.csv", sep=',')
#svd_test_nn_df = pd.read_csv("nn_prepared_svd_test.csv", sep=',')



#basic nn architecture:
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

#used for batch training:
import torch.utils.data as Data


xy_train = np.loadtxt("train_nn.csv", delimiter = ',' ,skiprows=1, dtype= np.float64 )
xy_test = np.loadtxt("test_nn.csv", delimiter = ',', skiprows =1 ,dtype = np.float64)


#########################
#########################
#prepare data for training

x_test = torch.from_numpy(xy_test[:,1:-1])
y_test = torch.from_numpy(xy_test[:,[-1]])


x_test = Variable(x_test.float())
y_test = Variable(y_test.float())


x_validation = torch.from_numpy(xy_train[:1265,1:-1])
y_validation = torch.from_numpy(xy_train[:1265,[-1]])
x_train = torch.from_numpy(xy_train[1265:,1:-1])
y_train = torch.from_numpy(xy_train[1265:,[-1]])


x_validation = Variable(x_validation.float())
y_validation = Variable(y_validation.float())
x_train = Variable(x_train.float())
y_train = Variable(y_train.float())


#define our own nn:
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden) 
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict=torch.nn.Linear(n_hidden,1) 
    #forward:
    def forward(self,x):
        x=F.relu(self.hidden(x)) 
        x=self.hidden2(x)
        x=self.predict(x) 
        return x


#initialize the net:
n_features = 81
net = Net(81, 40, 1)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.03)
loss_func=torch.nn.MSELoss()
loss_func_MAe=torch.nn.L1Loss()



#batch training:
Train_BATCH_SIZE = 200
Validate_BATCH_SIZE = 126

x_validation = torch.from_numpy(xy_train[:1260,1:-1])
y_validation = torch.from_numpy(xy_train[:1260,[-1]])
x_train = torch.from_numpy(xy_train[1260:,1:-1])
y_train = torch.from_numpy(xy_train[1260:,[-1]])

torch_dataset_train = Data.TensorDataset(data_tensor = x_train, target_tensor = y_train)
train_loader = Data.DataLoader(
    dataset = torch_dataset_train,
    batch_size = Train_BATCH_SIZE,
    shuffle = True,
    num_workers = 2,)

torch_dataset_validate = Data.TensorDataset(data_tensor = x_validation, target_tensor = y_validation)
validate_loader = Data.DataLoader(
    dataset = torch_dataset_validate,
    batch_size = Validate_BATCH_SIZE,
    shuffle = True,
    num_workers = 2,)


# In[232]:

#training:
#train too many steps will leads to overfit
#how to determine what is the stopping criterion of the training:
#---- one solution is using validation error
validation_loss = []
for epoch in range(100):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        print('Epoch:', epoch, '| Step: ', step)
        
        b_x = Variable(batch_x.float())
        b_y = Variable(batch_y.float())
        
        prediction = net(b_x)
        train_loss = loss_func(prediction, b_y)
        print('training loss is: ',train_loss.data[0])
        print('\n')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        #validation loss:
        if (step+1) % 13 ==0:
            sum_validation_loss = 0
            count = 0
            for s, (batch_x_validate, batch_y_validate) in enumerate(validate_loader):
                batch_x_validate = Variable(batch_x_validate.float())
                batch_y_validate = Variable(batch_y_validate.float())
                validation_output = net(batch_x_validate)
                #print(loss_func(validation_output, batch_y_validate).data[0])
                sum_validation_loss += loss_func(validation_output, batch_y_validate).data[0]
                count+=1
            validation_loss.append(sum_validation_loss/count)
            print("validation loss is {}".format(validation_loss[-1]))
            print('\n')
            print('\n')

            

#find where the validation error is the at minimum:
np.argmin(validation_loss)


#so the optimal training steps are 98* 13

#final model for nn:
count = 0
for epoch in range(100):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        print('Epoch:', epoch, '| Step: ', step)
        
        b_x = Variable(batch_x.float())
        b_y = Variable(batch_y.float())
        
        prediction = net(b_x)
        train_loss = loss_func(prediction, b_y)
        print('training loss is: ',train_loss.data[0])
        print('\n')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if count+1 == 98 * 13:
            break
        
        count +=1
        


#final prediction
final_prediction_y = net(x_test).data.numpy()
true_y = y_test.data.numpy()
plt.scatter(true_y, final_prediction_y)

plt.xlabel('predicted difference in sentencing length')
plt.ylabel('actual difference in sentencing length')
plt.show()


#final result avoid overfitting
print('mean squared error is: {}'.format(mean_squared_error(true_y, final_prediction_y)))
print('mean absolute error is {}'.format(mean_absolute_error(true_y, final_prediction_y)))



#compare Neural Network result with GBR with its best parameters performance

gbr = GradientBoostingRegressor()
parameters = {'n_estimators': [10, 50, 100, 200, 300], 'max_depth':[2,3,4,5], 'max_features':['auto','sqrt','log2']}
gbr_cv = GridSearchCV(gbr, parameters)
gbr_cv.fit(svd_train_df.iloc[:,:-1], svd_train_df.iloc[:,-1])

#best parameters
gbr_cv.best_params_

gbr_opt = GradientBoostingRegressor(max_depth= 2, max_features='log2', n_estimators= 10)
gbr_opt.fit(svd_train_df.iloc[:,:-1], svd_train_df.iloc[:,-1])

gbr_opt_tfidf_2gram_pred = gbr_opt.predict(svd_test_df.iloc[:,:-1])
gbr_opt_tfidf_2gram_mse = mean_squared_error(svd_test_df.iloc[:,-1], gbr_opt_tfidf_2gram_pred)
gbr_opt_tfidf_2gram_mae = mean_absolute_error(svd_test_df.iloc[:,-1], gbr_opt_tfidf_2gram_pred)

print('mean squared error is {}'.format(gbr_opt_tfidf_2gram_mse))
print('mean absolute error is {}'.format(gbr_opt_tfidf_2gram_mae))



### Look at R^2:

inference_data = pd.read_csv('bio_txt.csv')

true_df = inference_data[['index','0', '1',
       '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
       '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24','Res_binary', 'length_3m_dif']]


predicted_df = inference_data[['index','0_hat', '1_hat', '2_hat', '3_hat',
       '4_hat', '5_hat', '6_hat', '7_hat', '8_hat', '9_hat', '10_hat',
       '11_hat', '12_hat', '13_hat', '14_hat', '15_hat', '16_hat',
       '17_hat', '18_hat', '19_hat', '20_hat', '21_hat', '22_hat',
       '23_hat', '24_hat','Res_binary','length_3m_dif']]


predicted_df = predicted_df.drop(['index'],axis=1)
true_df = true_df.drop(['index'],axis=1)



#prepare data:
msk = np.random.rand(7388) < 0.8

predicted_df_x = predicted_df.iloc[:,:-1]
predicted_df_y = predicted_df.iloc[:,-1]

predicted_df_train_x = predicted_df_x[msk]
predicted_df_test_x = predicted_df_x[~msk]

predicted_df_train_y = predicted_df_y[msk]
predicted_df_test_y = predicted_df_y[~msk]


true_df_x = true_df.iloc[:,:-1]
true_df_y = true_df.iloc[:,-1]

true_df_train_x = true_df_x[msk]
true_df_test_x = true_df_x[~msk]

true_df_train_y = true_df_y[msk]
true_df_test_y = true_df_y[~msk]



#random forest regressor:

#predicted values R2 score
rfr_predicted_r2 = RandomForestRegressor(n_estimators=1000)
rfr_predicted_r2.fit(predicted_df_train_x, predicted_df_train_y)
rfr_predicted_r2_pred = rfr_predicted_r2.predict(predicted_df_test_x)

rfr_predicted_r2score = r2_score(predicted_df_test_y,rfr_predicted_r2_pred)
print('R^2 using randomforest regressor for predicted 25 dimension values: ',rfr_predicted_r2score)


#True values R2 score
rfr_true_r2 = RandomForestRegressor(n_estimators=1000)
rfr_true_r2.fit(predicted_df_train_x, predicted_df_train_y)
rfr_true_r2_pred = rfr_true_r2.predict(predicted_df_test_x)

rfr_true_r2score = r2_score(predicted_df_test_y,rfr_true_r2_pred)
print('R^2 using randomforest regressor for true 25 dimension values: ', rfr_true_r2score)



#Linear regression

#predicted value R2 score
lr_predicted_r2= linear_model.LinearRegression()
lr_predicted_r2.fit(predicted_df_train_x, predicted_df_train_y)
lr_predicted_r2_pred = lr_predicted_r2.predict(predicted_df_test_x)

lr_predicted_r2score = r2_score(predicted_df_test_y,lr_predicted_r2_pred)
print('R^2 using linear regression for predicted 25 dimension values: ', lr_predicted_r2score)


#True values R2 score
lr_true_r2= linear_model.LinearRegression()
lr_true_r2.fit(true_df_train_x, true_df_train_y)
lr_true_r2_pred = lr_true_r2.predict(true_df_test_x)

lr_true_r2score = r2_score(true_df_test_y,lr_true_r2_pred)
print('R^2 using randomforest regressor for true 25 dimension values: ', lr_true_r2score)



