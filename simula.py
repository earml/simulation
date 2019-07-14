# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:18:13 2019

@author: elisalvo
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') # = rm(list = ls()) or
%reset -f
import pandas as pd
import numpy as np
import psutil
import time
import random
from IPython.display import HTML
from sklearn.linear_model import LogisticRegression
#import statsmodels as sm
#import statsmodels.formula.api as sm
import statsmodels.api as sm
import seaborn as sns # For plotting/checking assumptions
import gc

np.random.seed(seed = 123456789)

d = pd.read_csv("E:\\Simulation\\base6\\base6.csv", sep = "|")

d_simulated = d

sample_0 = d_simulated[d_simulated.pesocat == 0]
sample_1 = d_simulated[d_simulated.pesocat == 1]

base_fold = list()

dim_1 = [20891, 12794, 8745, 6316, 4696, 3539]
fold_0 = [2, 3, 5, 7, 9, 13]

nCores = psutil.cpu_count(logical = False)
nThreads = psutil.cpu_count(logical = True)
print("CPU with", nCores, "cores and",nThreads,"threads detected.\n")

start = time.time()

for init in dim_1:
    
    var_base_simu_final = list(); sintese_final = list()
    
    print (init,' dimensao :))')
    
    if init == 20891:
        fold_00 = 2
    elif init == 12794:
        fold_00 = 3
    elif init == 8745:
        fold_00 = 5
    elif init == 6316:
        fold_00 = 7
    elif init == 4696:
        fold_00 = 9
    elif init == 3539:
        fold_00 = 13
    print (fold_00, 'amostras')
    
    for j in range(0, 1000):
        start_1 = 0; start_2 = 0; start_3 = init
        var_base_simu = []; sintese = list()
        seq = np.arange(1, len(sample_0) + 1, 1).tolist()
        ind = random.sample(seq, len(seq))# Gero os indices sem reposicao para gerar as bases disjuntas
        
        for i in range(start_1, fold_00):
            print (j, i)
            base_1 = sample_1.append(sample_0.iloc[ind[start_2:start_3],:])
            var_base_simu.append(base_1)
                        
            base_1 = pd.get_dummies(base_1, columns = ['escmae', 'racacorn'], drop_first = True)
            
            base_1['Intercept'] = 1
            model_logit = sm.Logit(base_1.pesocat, base_1.iloc[:,1:base_1.shape[1]]).fit()
            
            out2 = model_logit.summary2()
            coef = pd.DataFrame(out2.tables[1].loc[:,['Coef.','Std.Err.','P>|z|']]).reset_index(drop = True)
            interval_conf = pd.DataFrame(out2.tables[1].loc[:,['[0.025','0.975]']]).reset_index(drop = True)
            odds_r = pd.DataFrame(np.exp(interval_conf))
            
            var_indep = pd.DataFrame(list(out2.tables[1].index))
            var_indep.columns = ['Var']
            probY = pd.DataFrame(np.repeat(i,var_indep.size))
            probY.columns = ['probY']
                
            result = pd.concat([var_indep,probY, coef[['Coef.']],interval_conf,odds_r], axis = 1)                
            result.columns = ['Var', 'probY', 'Coef.', 'IC_2.5', 'IC_97.5', 'odds_2.5', 'odds_97.5']         
            
            sintese.append(result)
            
            start_2 = start_2 + init
            start_3 = start_3 + init 
      
        matrix_base_simu = var_base_simu
        result_base_simu = sintese
        ###
                
        for s in range(0, len(result_base_simu)):
            result = result_base_simu[s]
            if s == 0:
                d = pd.concat([result])
            elif s > 0:
                d = pd.concat([d,result])
        
        for t in range(0, len(matrix_base_simu)):
            result2 = matrix_base_simu[t]
            if t == 0:
                d2 = pd.concat([result2])
            elif t > 0:
                d2 = pd.concat([d2,result2])
            
            
        #np.hstack(sintese)
        #np.stack(sintese)
        ###
        var_base_simu_final.append(d2)
        sintese_final.append(d)
        gc.collect()

 
    for u in range(0, len(var_base_simu_final)):
            result3 = var_base_simu_final[u]
            if u == 0:
                d3 = pd.concat([result3])
            elif u > 0:
                d3 = pd.concat([d3,result3])
    
    for v in range(0, len(sintese_final)):
            result4 = sintese_final[v]
            if v == 0:
                d4 = pd.concat([result4])
            elif v > 0:
                d4 = pd.concat([d4,result4])
                
    #matrix_final = np.matrix(var_base_simu_final)
    #matrix_final = np.ravel(var_base_simu_final)
    matrix_final = d3
    base_final = d4
    
    pd.concat(matrix_final).to_csv("E:/python/base_simula{}".format(init)+".csv", sep = "|",decimal = ".",header = True)
    base_final.to_csv("E:/python/base_final{}".format(init)+".csv", sep = "|",decimal = ".", header = True)
    del(matrix_base_simu); del(result_base_simu)
    del(sintese_final); del(var_base_simu_final)
    
    gc.collect()
        
        
(time.time() - start)


