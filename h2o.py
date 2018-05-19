
# coding: utf-8

# In[1]:


import h2o
import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


# ## Load data and initialize h2o

# In[2]:


with open('data/cc_merged_0516.pkl', 'rb') as f:
    rawdata = pickle.load(f)


# In[3]:


rawdata.head()


# In[4]:


h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()                          #clean slate, in case cluster was already running


# In[5]:


rawdata = h2o.H2OFrame(python_obj=rawdata)


# In[6]:


rawdata.shape


# In[7]:


rawdata.head()


# In[106]:


#h2o groupby example
rawdata.group_by("opinion_type").sum('Affirmed').get_frame()


# In[9]:


rawdata.col_names


# In[10]:


# de-mean for x_dem
# rawdata[:,"x_dem"] -= rawdata[:,"x_dem_p"]


# In[11]:


# rawdata[:,"x_dem_y"] -= rawdata[:,"x_dem_p"]
# rawdata[:,"x_dem_x"] -= rawdata[:,"x_dem_p"]


# ## Generate target variable

# In[12]:


data = rawdata[sum(rawdata[:,["Affirmed","AffirmedInPart","Reversed","ReversedInPart","Vacated","VacatedInPart"]]) == 1,:]#.sum(axis=0)
data["res"] = 0

# data = data.as_data_frame(use_pandas=True)
# data = data[data["res"] == 1,:] 
# data = h2o.H2OFrame(data)
data[data["Affirmed"] == 1, "res"] = 1
data[data["Reversed"] == 1, "res"] = 2
data[data["Vacated"] == 1, "res"] = 2
# # data_r = data[data["res"] == "Reversed"]
# # data_a = data[data["res"] == "Affirmed"]
# # data = data_r.append(data_a)


# In[13]:


# data = data.drop("x_dem_p", 1)


# In[14]:


data_bio = data[:,["index"]+[c for c in data.columns if c[:2] == "x_"]].cbind(data["res"])


# In[15]:


data_bio.col_names[213]


# ## Sum 3 Judges

# ### Random Forest

# In[16]:


#sum up three judges
for i in range(143, 213 + 1):
    data_bio[:,i] += data_bio[:,i - 71] + data_bio[:,i - 142]
data_bio_sumed = data_bio["index"].cbind(data_bio[:,143:])
data_bio_sumed.show()


# In[17]:


#modelling
#X : bio_data_sumed
#y : data["res"] == 1 or 2
filt = data_bio_sumed["res"] != 0
data_bio_sumed_filt = data_bio_sumed[filt]
data_bio_sumed_filt["res"] = data_bio_sumed_filt["res"].asfactor()


# In[18]:


#train_test_split
train, test, val = data_bio_sumed_filt.split_frame(ratios=[.7, .15])


# In[19]:


#fit a random forest
rf_v1 = H2ORandomForestEstimator(
    model_id="rf_v1",
    ntrees=120,
    max_depth = 6,
#     stopping_rounds=2,
    score_each_iteration=True,
    seed=10000)


# In[20]:


train_X = train.col_names[1:-1]     #last column is Cover_Type, our desired response variable 
train_y = train.col_names[-1]
rf_v1.train(train_X, train_y, training_frame=train, validation_frame=val)


# In[21]:


pred = rf_v1.predict(val[:,1:-1]).as_data_frame().as_matrix()[:,-2:].ravel()
true = pd.get_dummies(val[:,-1].as_data_frame().as_matrix().flatten()).values.ravel()
print("AUC Score calculaed by sklearn")
roc_auc_score(true, pred)


# In[67]:


plt.rcdefaults()
fig, ax = plt.subplots()
variables = rf_v1._model_json['output']['variable_importances']['variable']
y_pos = np.arange(10)
scaled_importance = rf_v1._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance[:10], align='center', color='skyblue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.savefig('imp.png')
plt.show()

rf_v1._model_json['output']['variable_importances'].as_data_frame()


# In[27]:


perf = rf_v1.model_performance(valid = True)
perf.plot()


# In[69]:


rf_v1.r2()


# ### Gradient Boost

# In[61]:


gbm1 = H2OGradientBoostingEstimator()
gbm1.train(train_X, train_y, training_frame=train, validation_frame=val)


# In[62]:


pred = gbm1.predict(val[:,1:-1]).as_data_frame().as_matrix()[:,-2:].ravel()
true = pd.get_dummies(val[:,-1].as_data_frame().as_matrix().flatten()).values.ravel()
print("AUC Score calculaed by sklearn")
roc_auc_score(true, pred)


# In[63]:


gbm1.confusion_matrix(valid=True)


# In[41]:


gbm1.model_performance(valid=True)


# In[42]:


plt.rcdefaults()
fig, ax = plt.subplots()
variables = gbm1._model_json['output']['variable_importances']['variable']
y_pos = np.arange(10)
scaled_importance = gbm1._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance[:10], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()

gbm1._model_json['output']['variable_importances'].as_data_frame()


# In[ ]:


## sort the grid models by decreasing AUC
sorted_grid = grid.get_grid(sort_by='auc',decreasing=True)
print(sorted_grid)


# ### Top 10 -> Logistics Regression

# In[58]:


top_fea = ["x_agecommi", "x_republican", "x_dem", "x_jewish", "x_hrep", "x_srep", "x_hdem", "x_sdem", "x_aba", "x_catholic"]
lr_v1 = H2OGeneralizedLinearEstimator(
                    model_id='lr1',            #allows us to easily locate this model in Flow
                    family='binomial',
                    solver='L_BFGS')
lr_v1.train(top_fea, train_y, training_frame = train, validation_frame = val)


# In[59]:


lr_v1.coef_norm()


# In[25]:


help(H2OGeneralizedLinearEstimator)


# ## Un-sum 3 Judges

# ### Random Forsest

# In[223]:


#modelling using only un-summed biodata
#fit a random forest
rf_v2 = H2ORandomForestEstimator(
    model_id="rf_v2",
    ntrees=120,
    max_depth = 6,
#     stopping_rounds=2,
    score_each_iteration=True,
    seed=10000)
data_bio_filt = data_bio[data_bio["res"] != 0]
data_bio_filt["res"] = data_bio_filt["res"].asfactor()
train, test, val = data_bio_filt.split_frame(ratios=[.7, .15])
train_X = train.col_names[:-1]     #last column is Cover_Type, our desired response variable 
train_y = train.col_names[-1]
rf_v2.train(train_X, train_y, training_frame=train, validation_frame=val)


# In[240]:


rf_v2.model_performance()


# In[225]:


pred = rf_v2.predict(val[:,:-1]).as_data_frame().as_matrix()[:,-2:].ravel()
true = pd.get_dummies(val[:,-1].as_data_frame().as_matrix().flatten()).values.ravel()
roc_auc_score(true, pred)


# In[242]:


rf_v2.confusion_matrix(valid=True)


# ## Generalize

# In[70]:


data_bio_sumed_filt.shape


# In[71]:


all_pred = rf_v1.predict(data_bio_sumed_filt[:,1:-1]).as_data_frame().as_matrix()[:,-2:]


# In[72]:


rf_v1.confusion_matrix(valid=True)


# In[75]:


deci = [ls[1] > 0.17094308502972125 for ls in all_pred]


# In[76]:


sum(deci)


# In[77]:


data_bio_sumed_filt[:,"pred"] = h2o.H2OFrame(deci)


# In[78]:


data_bio_sumed_filt = data_bio_sumed_filt[:,["index","pred","res"]]
data_bio_sumed_filt.summary()


# In[79]:


data_out = rawdata.merge(data_bio_sumed_filt, all_x=True, by_x=["index"], by_y=["index"])


# In[80]:


data_out = data_out.drop("txt", 1)


# In[81]:


data_out = data_out.as_data_frame(use_pandas=True)


# In[82]:


data_out.ix[data_out["pred"] == True, "pred"] = "Rev"


# In[83]:


data_out.ix[data_out["pred"] == False, "pred"] = "Aff"


# In[84]:


data_out.pred.value_counts()


# In[85]:


data_out


# In[86]:


data_out.to_csv("data_bio_sumed_pred.csv")


# ## Judge -> Text Features

# In[87]:


pt1 = pd.read_csv("data/nn_prepared_svd_test.csv", header=0, index_col=0)


# In[88]:


pt2 = pd.read_csv("data/nn_prepared_svd_train.csv", header=0, index_col=0)


# In[89]:


pt = pt1.append(pt2)


# In[90]:


pt.head()


# In[116]:


data_bio_sumed_filt_pd = data_bio_sumed_filt.as_data_frame(use_pandas=True)


# In[117]:


bio_txt = pd.merge(data_bio_sumed_filt_pd, pt, left_on = "index", right_on = "index")


# In[118]:


bio_txt.columns


# In[119]:


y_s = ['0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
       '18', '19', '20', '21', '22', '23', '24']


# In[120]:


x_col = ['x_dem_dm', 'x_republican_dm', 'x_instate_ba_dm', 'x_elev_dm',
       'x_unity_dm', 'x_aba_dm', 'x_crossa_dm', 'x_pfedjdge_dm',
       'x_pindreg1_dm', 'x_plawprof_dm', 'x_pscab_dm', 'x_pcab_dm',
       'x_pusa_dm', 'x_pssenate_dm', 'x_paag_dm', 'x_psp_dm', 'x_pslc_dm',
       'x_pssc_dm', 'x_pshouse_dm', 'x_psg_dm', 'x_psgo_dm', 'x_psenate_dm',
       'x_psatty_dm', 'x_pprivate_dm', 'x_pmayor_dm', 'x_plocct_dm',
       'x_phouse_dm', 'x_pgov_dm', 'x_pda_dm', 'x_pcc_dm', 'x_pccoun_dm',
       'x_pausa_dm', 'x_pasatty_dm', 'x_pag_dm', 'x_pada_dm', 'x_pgovt_dm',
       'x_llm_sjd_dm', 'x_protestant_dm', 'x_evangelical_dm', 'x_mainline_dm',
       'x_noreligion_dm', 'x_catholic_dm', 'x_jewish_dm', 'x_black_dm',
       'x_nonwhite_dm', 'x_female_dm', 'x_jd_public_dm', 'x_ba_public_dm',
       'x_b10s_dm', 'x_b20s_dm', 'x_b30s_dm', 'x_b40s_dm', 'x_b50s_dm',
       'x_pbank_dm', 'x_pmag_dm', 'x_ageon40s_dm', 'x_ageon50s_dm',
       'x_ageon60s_dm', 'x_ageon40orless_dm', 'x_ageon70ormore_dm',
       'x_pago_dm', 'x_apptoter_dm', 'x_term_dm', 'x_circuit_dm', 'x_hdem_dm',
       'x_hrep_dm', 'x_sdem_dm', 'x_srep_dm', 'x_hother_dm', 'x_sother_dm',
       'x_agecommi_dm']


# In[121]:


bio_txt_h = h2o.H2OFrame(python_obj=bio_txt)


# In[122]:


train, val = bio_txt_h.split_frame(ratios=[.7])


# In[123]:


pred = {}
r2 = {}

for y in y_s:
    print("now training :", y)
    train_X = x_col     #last column is Cover_Type, our desired response variable 
    train_y = y
    rf_v1.train(train_X, train_y, training_frame=train, validation_frame=val)
    print(rf_v1.r2(valid=True))
    yhat_tot = rf_v1.predict(bio_txt_h[:,x_col])
    bio_txt_h[y + "_hat"] = yhat_tot
    r2[y] = rf_v1.r2(valid=True)
    print("*****************")
#     yhat = regr.predict(X_val)
#     r2 = r2_score(y_test[y], yhat)
#     print("r2 on val set", r2)
#     
#     res[y] = yhat_tot
#     print("*****************")


# In[124]:


bio_txt_h.show()


# In[125]:


bio_txt_df = bio_txt_h.as_data_frame(use_pandas=True)


# In[126]:


bio_txt_df.to_csv("bio_txt.csv")


# In[127]:


r2_df = pd.DataFrame([r2])
r2_df.to_csv("r2_dm.csv")

