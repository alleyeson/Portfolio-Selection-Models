#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:46:49 2019

@author: allisonemono
"""

## start with simple markowitiz 
def marko(data):
    import pandas as pd
    import numpy as np
    #import matplotlib.pyplot as plt
    
    sig_inv_m = np.cov(np.transpose(data))
    sig_inv_m = np.linalg.inv(sig_inv_m)
    
    N_mshape = np.shape(sig_inv_m)
    N_m = N_mshape[1]
    ones = np.array(np.repeat(1,N_m))
    left = np.matmul(sig_inv_m,ones)
    lam = np.dot(ones,left)
    
    x = (1/lam)* left
    
    return x


def marko_w_target_ret_f(data,targ_mu):
    import pandas as pd
    import numpy as np
    
    #p = np.shape(data)[1]
    mu_v = data.mean(axis = 0) 
    sig_inv_m = np.cov(np.transpose(data))
    sig_inv_m = np.linalg.inv(sig_inv_m)
    N_mshape = np.shape(sig_inv_m)
    p = N_mshape[1]
    
    B_temp = np.matmul(sig_inv_m,np.transpose(mu_v) )
    b = np.dot(mu_v,B_temp)
    ones = np.array(np.repeat(1,p))
    a = np.dot(ones,B_temp)
    c =  np.dot(ones, np.matmul(sig_inv_m, ones))
    d = b*c - a*a
    
    target_returns = targ_mu 
    cd =c/d
    ad = a/d
    bd = b/d
    
    muCoeff = np.array(cd*target_returns - ad)
    oneCoeff = np.array(bd - ad*target_returns)
    w = np.matmul(sig_inv_m, np.transpose(muCoeff * mu_v + oneCoeff*ones) )
    w = np.array(w)

    
    
    return w

def sharpe_marko_f(data):
    import pandas as pd
    import numpy as np
    
    mu_v = np.transpose(data.mean(axis = 0))
    var_inv_m = np.linalg.inv(np.cov(np.transpose(data)) )

    d = np.shape(data)[1]
    ones_v = np.transpose(np.matrix(np.repeat(1,d)) )
    
    left_v = mu_v - ones_v
    denum = np.dot( np.transpose(ones_v), np.matmul(np.transpose(var_inv_m),(left_v)) )
    
    j = np.matmul(var_inv_m, left_v)
    j = j*(1/denum)
    
    return j


data_m = pd.read_csv('/Users/allisonemono/Desktop/Python Session/518StockData.csv')

type(data_m)


df= pd.DataFrame(data_m)
type(df)
df.head
df.columns
df_data_m  = df.drop('Date', axis=1)

df_data_m.columns

## now to have data into matrix of sorts 
data = np.matrix(df_data_m)

## now check if portfolio obeys restrictions
print(sum(marko(data)))
np.shape(marko(data))

sum(marko_w_target_ret_f(data,0.5))
np.shape(marko_w_target_ret_f(data,0.5))

print(sum(sharpe_marko_f(data)))