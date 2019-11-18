#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:34:27 2019

@author: sizhenhan
"""

import pandas as pd
import numpy as np

def frequency_encode(df1, df2, cols):
    for col in cols:
        whole_list = pd.concat([df1[col],df2[col]])
        proportion = whole_list.value_counts(dropna=True, normalize=True).to_dict()
        name = col+'_FE'
        df1[name] = df1[col].map(proportion)
        df1[name] = df1[name].astype('float32')
        df2[name] = df2[col].map(proportion)
        df2[name] = df2[name].astype('float32')
        

def label_encode(col,train,test):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    train[col] = df_comb[:len(train)].astype('int32')
    test[col] = df_comb[len(train):].astype('int32')

def combine(col1,col2,df1,df2):
    name = col1+'_'+col2
    df1[name] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[name] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    label_encode(name,df1,df2)
    
def aggregate_encode(column1s, column2s, aggregations,train_df, test_df):
    for col1 in column1s:  
        for col2 in column2s:
            for agg in aggregations:
                new_name = col1 + '_' + col2 + '_' + agg
                df = pd.concat([train_df[[col2, col1]], test_df[[col2,col1]]])
                df.loc[df[col1]==-1,col1] = np.nan
                df = df.groupby([col2])[col1].agg([agg]).reset_index()
                df = df.rename(columns={agg: new_name})

                df.index = list(df[col2])
                df = df[new_name].to_dict()   

                train_df[new_name] = train_df[col2].map(df).astype('float32')
                test_df[new_name]  = test_df[col2].map(df).astype('float32')
                
                train_df[new_name].fillna(-1,inplace=True)
                test_df[new_name].fillna(-1,inplace=True)