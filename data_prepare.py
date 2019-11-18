#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:58:10 2019

@author: sizhenhan
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data,data_id,index_cols = 'TransactionID'):
    
    data = pd.read_csv(data,index_col = index_cols)
    data_id = pd.read_csv(data_id,index_col = index_cols)
    data = data.merge(data_id, how='left', left_index=True, right_index=True)
    print(data.head())
    y = data['isFraud']
    x = data.drop(['isFraud'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2019)
    
    return X_train, X_test, y_train, y_test


def transforms(X_train_final,X_test_final):
    
    Ms = ['M'+str(x) for x in range(1,10)]
    Ms.remove('M5')
    mp = {'F':0,'T':1,'M0':0,'M1':1,'M2':2}
    for c in Ms: X_train_final[c] = X_train_final[c].map(mp)
    for c in Ms: X_test_final[c] = X_test_final[c].map(mp)
    
    l = []
    for i in range(len(X_train_final.columns)):
        if X_train_final[X_train_final.columns[i]].dtype == 'O':
            l.append(X_train_final.columns[i])
            
    for column in l:
        X_train_final[column] = pd.Categorical(X_train_final[column])
        X_train_final[column] = X_train_final[column].cat.codes
    
    for column in l:
        X_test_final[column] = pd.Categorical(X_test_final[column])
        X_test_final[column] = X_test_final[column].cat.codes
    
    X_train_final = X_train_final.replace({pd.np.nan: -1.0})
    X_test_final = X_test_final.replace({pd.np.nan: -1.0})
    
    return X_train_final, X_test_final

def feature_selection(X_train):
    
    remove_v = ['V3','V5','V7','V9',
 'V10',
 'V12',
 'V15',
 'V16',
 'V18',
 'V19',
 'V21',
 'V22',
 'V24',
 'V25',
 'V28',
 'V29',
 'V31',
 'V32',
 'V33',
 'V34',
 'V35',
 'V37',
 'V40',
 'V42',
 'V43',
 'V45',
 'V46',
 'V49',
 'V50',
 'V51',
 'V52',
 'V53',
 'V55',
 'V57',
 'V58',
 'V60',
 'V61',
 'V63',
 'V64',
 'V66',
 'V69',
 'V71',
 'V72',
 'V73',
 'V74',
 'V95',
 'V97',
 'V100',
 'V101',
 'V102',
 'V103',
 'V105',
 'V106',
 'V109',
 'V110',
 'V112',
 'V113',
 'V114',
 'V116',
 'V118',
 'V119',
 'V122',
 'V125',
 'V126',
 'V128',
 'V131',
 'V132',
 'V133',
 'V134',
 'V135',
 'V137',
 'V140',
 'V141',
 'V143',
 'V144',
 'V145',
 'V146',
 'V148',
 'V149',
 'V150',
 'V151',
 'V152',
 'V153',
 'V154',
 'V157',
 'V158',
 'V159',
 'V161',
 'V163',
 'V164',
 'V167',
 'V168',
 'V170',
 'V172',
 'V174',
 'V177',
 'V179',
 'V181',
 'V183',
 'V184',
 'V186',
 'V189',
 'V190',
 'V191',
 'V192',
 'V193',
 'V194',
 'V195',
 'V196',
 'V197',
 'V199',
 'V200',
 'V201',
 'V202',
 'V204',
 'V206',
 'V208',
 'V211',
 'V212',
 'V213',
 'V214',
 'V216',
 'V217',
 'V219',
 'V225',
 'V230',
 'V231',
 'V232',
 'V233',
 'V236',
 'V237',
 'V241',
 'V242',
 'V243',
 'V244',
 'V246',
 'V247',
 'V248',
 'V249',
 'V254',
 'V262',
 'V263',
 'V265',
 'V268',
 'V269',
 'V273',
 'V275',
 'V276',
 'V278',
 'V279',
 'V280',
 'V282',
 'V287',
 'V288',
 'V290',
 'V292',
 'V293',
 'V295',
 'V298',
 'V299',
 'V300',
 'V302',
 'V304',
 'V306',
 'V308',
 'V311',
 'V312',
 'V313',
 'V315',
 'V316',
 'V317',
 'V318',
 'V319',
 'V321',
 'V322',
 'V323',
 'V324',
 'V326',
 'V327',
 'V328',
 'V329',
 'V330',
 'V331',
 'V333',
 'V334',
 'V336',
 'V337',
 'V339']
    
    cols = list(X_train.columns)
    cols.remove('TransactionDT')

    for c in remove_v:
        cols.remove(c)

    for c in ['D6','D7','D8','D9','D12','D13','D14']:
        cols.remove(c)
    
    for c in ['C3','M5','id_08','id_33']:
        cols.remove(c)
        
    for c in ['card4','id_07','id_14','id_21','id_30','id_32','id_34']:
        cols.remove(c)
        
    for c in ['id_'+str(x) for x in range(22,28)]:
        cols.remove(c)
    
    return cols
    