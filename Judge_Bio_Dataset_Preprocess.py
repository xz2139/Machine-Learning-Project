
# coding: utf-8

import pandas as pd

##read judge characteristics
d=pd.read_stata('/data/Dropbox/Data/Judge-Bios/judgebios/judgebios_circuit.dta')
pd.get_dummies(d.abarating)

## convert time to days
d['AppointmentDate'] = pd.to_datetime(d['AppointmentDate'])
d['TerminationDate'] = pd.to_datetime(d['TerminationDate'])
d['day_Appoint_to_Terminat']=(d['TerminationDate'] - d['AppointmentDate']).dt.days

##convert to categorical features
d=d.join(pd.get_dummies(d.courtname))


d=d.drop(['courttype'],axis=1)

b=pd.read_stata('/data/Dropbox/Data/Judge-Bios/judgebios/JudgesBioReshaped_TOUSE.dta')


s=[col for col in d if col.startswith('x_')]
d_x=d[s]

##rename the features
d_x['x_apptoter']=d.day_Appoint_to_Terminat
d_x['x_dccircuit']=d['u. s. court of appeals for the district of columbia circuit']
d_x['x_dccircuit']+=d['u. s. court of appeals for the district of columbia circuit, chief justice']
d_x['x_8circuit']=d['u. s. court of appeals for the eighth circuit']
d_x['x_11circuit']=d['u. s. court of appeals for the eleventh circuit']
d_x['x_fcircuit']=d['u. s. court of appeals for the federal circuit']
d_x['x_5circuit']=d['u. s. court of appeals for the fifth circuit']
d_x['x_1circuit']=d['u. s. court of appeals for the first circuit']
d_x['x_4circuit']=d['u. s. court of appeals for the fourth circuit']
d_x['x_9circuit']=d['u. s. court of appeals for the ninth circuit']
d_x['x_2circuit']=d['u. s. court of appeals for the seventh circuit']
d_x['x_7circuit']=d['u. s. court of appeals for the second circuit']
d_x['x_6circuit']=d['u. s. court of appeals for the sixth circuit']
d_x['x_10circuit']=d['u. s. court of appeals for the tenth circuit']
d_x['x_3circuit']=d['u. s. court of appeals for the third circuit']

d_x['id']=d.judgeidentificationnumber
d_x['x_term']=d.Term
d_x['x_circuit']=d.Circuitjudge
d_x['x_aba']=d.aba
d_x['x_hdem']=d.hdem
d_x['x_hrep']=d.hrep
d_x['x_sdem']=d.sdem
d_x['x_srep']=d.srep
d_x['x_hother']=d.hother
d_x['x_sother']=d.sother
d_x['x_agecommi']=d.ageon

d_x['fullname']=d.songername
d_x['lastname']=d.judgelastname
d_x['firstname']=d.judgefirstname
d_x['middlename']=d.judgemiddlename
d_x['suffix']=d.suffix
d_x['x_appres']=d.appres

##save to file
d_x.to_csv('JudgeBio_x_name.csv',index=False)

