
# coding: utf-8

###unzip sentences text first from /data/Dropbox/Data/Circuit_Courts/circuit-cases/sentences.zip


from zipfile import ZipFile 
from random import shuffle
zfile = ZipFile('/data/Dropbox/Data/Circuit_Courts/circuit-cases/sentences.zip')
items = zfile.namelist()

shuffle(items)
for count, item in enumerate(items):`
    _,_year,fname = item.split('/')
    _, year = _year.split('_')
    if not count % 50000:
      print('%d files processed' % count)
    txt = zfile.open(item).read().decode()
    # run model here



import datetime
import pandas as pd
import pickle
import glob

dd=pd.read_stata('/data/Dropbox/Data/District_Court_Sentencing/sentencing_since_1991.dta')


def combinations(line):
    l = line.strip().split(',')
    combs=[]        
    for i in range(2, len(l)+1):
        els = [",".join(x) for x in itertools.combinations(l, i)]
        combs.extend(els)
    return combs



newdd=dd[['dateofsentencing','district','lengthofprisonsentence']]



newdd=newdd[newdd.dateofsentencing.isnull()==False]


newdd.dateofsentencing[newdd.dateofsentencing.astype(int)>500000]+=19000000


newdd.dateofsentencing[newdd.dateofsentencing.astype(int)<=500000]+=20000000


newdd['Date']=newdd.dateofsentencing.astype(int).astype(str)


newdd['date']=pd.to_datetime(newdd['Date'])



def convertlength(df):
    orig_leng = df["lengthofprisonsentence"]
    if len(orig_leng) < 2:
        return None
    elif orig_leng[-1] == "Y":
        return int(orig_leng[:-1])*12
    elif orig_leng[-1] == "M":
        return int(orig_leng[:-1])
    else:
        return None
newdd["lengthofprisonsentence"] = newdd.apply(convertlength, axis=1)



newdd=newdd[newdd.lengthofprisonsentence.isnull()==False]
newdd[['district','lengthofprisonsentence','date']]



k=pd.read_stata('/data/Dropbox/Projects/originalism/data/BloombergVOTELEVEL_Touse.dta')
k=k[k.geniss1==1] ##filter on criminal cases
meta=k[['date','JudgesListTouse','geniss1','Author','songername','caseid','Affirmed','AffirmedInPart','Reversed','ReversedInPart','Vacated','VacatedInPart','Circuit','judgelastname','jOrigname','judgeidentificationnumber','yearq']]


meta.to_pickle('metadata.pkl')
meta=pickle.load(open("metadata.pkl","rb"))



group=meta.groupby('caseid', as_index=False)['judgeidentificationnumber'].agg({'list':(lambda x: list(x))})



merge1=pd.merge(meta, group, how='inner', left_on='caseid', right_on='caseid')



merge1.rename(columns={'judgeidentificationnumber': 'judgeid1'}, inplace=True)



merge1['judgeid1']=[item[0] for item in merge1.list]
merge1['judgeid2']=[item[1] for item in merge1.list]
merge1['judgeid3']=[item[2] for item in merge1.list]



merge1=merge1.drop_duplicates('caseid')



meta=merge1.drop(['JudgesListTouse','geniss1','songername','jOrigname','judgelastname','list','yearq'],axis=1)


cc=pickle.load( open( "Sentencing/sentence_data.csv", "rb" ) )



bio=pd.read_csv('Bio/JudgeBio_x_name.csv')




merge1=pd.merge(cc, meta, how='inner', left_on='caseid', right_on='caseid')


merge2=pd.merge(merge1, bio, how='inner', left_on='judgeid1', right_on='id')



merge3=pd.merge(merge2, bio, how='inner', left_on='judgeid2', right_on='id')



merge4=pd.merge(merge3, bio, how='inner', left_on='judgeid3', right_on='id')

count=k.groupby(['year'])['x_dem'].count()
sum=k.groupby(['year'])['x_dem'].sum()

def dem(year):
    return dict(sum/count)[float(year)]

merge4['x_dem_p']=merge4.year.apply(dem)


cc=merge4
dc=newdd
cc['month_3']=cc.date+pd.DateOffset(months=3)
cc['month_3_b']=cc.date+pd.DateOffset(months=-3)

def find_date(df):
    min_date = df["month_3_b"]
    date=df["date"]
    max_date = df["month_3"]
    a=dc[(dc['date'] >= date) & (dc['date'] <= max_date)].lengthofprisonsentence.mean()
    b=dc[(dc['date'] >= min_date) & (dc['date'] <= date)].lengthofprisonsentence.mean()

    return a-b

cc["length_3m_dif"] = cc.apply(find_date, axis=1)


s=cc.dropna(subset = ['length_3m_dif'])


import nltk
from nltk.util import ngrams
from io import open
import unicodedata
import string
import re


s['txt']=None



for i,(year,ids,j) in enumerate(zip(s.year,s.caseid,s.judge_name)):
    filepath = "Sentencing/sentences_new/sent_"+str(year)+"/"+ids+"_*_"+j+"*.txt"
    txt = glob.glob(filepath)
    with open (txt[0],encoding="utf8") as f:
        #c=Counter(list(f.read()))
        text=f.read().lower()
        text = re.sub("\n", ' ', text)
        s.txt[i]=text


s.to_pickle('cc_merged_0412.pkl.pkl')

count=k.groupby(['year'])['x_dem'].count()
sum=k.groupby(['year'])['x_dem'].sum()

def dem(year):
    return dict(sum/count)[float(year)]

merge4['x_dem_p']=merge4.year.apply(dem)


cc=s

cc=cc.drop(['x_dccircuit_x',
 'x_8circuit_x',
 'x_11circuit_x',
 'x_fcircuit_x',
 'x_5circuit_x',
 'x_1circuit_x',
 'x_4circuit_x',
 'x_9circuit_x',
 'x_2circuit_x',
 'x_7circuit_x',
 'x_6circuit_x',
 'x_10circuit_x',
 'x_3circuit_x','x_dccircuit_y',
 'x_8circuit_y',
 'x_11circuit_y',
 'x_fcircuit_y',
 'x_5circuit_y',
 'x_1circuit_y',
 'x_4circuit_y',
 'x_9circuit_y',
 'x_2circuit_y',
 'x_7circuit_y',
 'x_6circuit_y',
 'x_10circuit_y',
 'x_3circuit_y','x_dccircuit',
 'x_8circuit',
 'x_11circuit',
 'x_fcircuit',
 'x_5circuit',
 'x_1circuit',
 'x_4circuit',
 'x_9circuit',
 'x_2circuit',
 'x_7circuit',
 'x_6circuit',
 'x_10circuit',
 'x_3circuit'],axis=1)


cols=[col for col in cc if col.startswith('x_')]

for i in cols:
    dictionary=cc.groupby(['year','x_circuit_x'])[i].mean().to_dict()
    def demean(df):
        circuit = df['x_circuit_x']
        year=df['year']
        before=df[i]
        mean=dictionary[(year,circuit)]
        return before-mean

    cc[i+'_dm'] = cc.apply(demean, axis=1)


cc=cc.drop(cols+['x_circuit_x','x_circuit_y','x_circuit'],axis=1)


cc.reset_index().to_pickle('cc_merged_0516.pkl')

