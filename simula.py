# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:18:13 2019

@author: elisalvo
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') # = rm(list = ls())
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


np.random.seed(seed = 123456789)

d = pd.read_csv("E:\\Rose\\Standard_menard\\Simulation\\base6\\base6.csv", sep = "|")

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
            coef = pd.DataFrame(out2.tables[1].loc[:,['Coef.','Std.Err.','P>|z|']])
            interval_conf = pd.DataFrame(out2.tables[1].loc[:,['[0.025','0.975]']])
            odds_r = pd.DataFrame(np.exp(interval_conf))
            '''var_indep = pd.DataFrame(np.concatenate(('Intercept', list(odds_r.index)), axis = None))'''
            var_indep = pd.DataFrame(list(odds_r.index))
            var_indep.columns = ['Var']
            probY = pd.DataFrame(np.repeat(i,var_indep.size))
            probY.columns = ['probY']
                
#######??????????????????????????????                                   
            res = pd.DataFrame()
            res[['Var']] = var_indep
            res[['probY']] = probY
            res[['Coef.']] = coef[['Coef.']]
                        
            res2 = pd.DataFrame()
            res2[['IC_2.5', 'IC_97.5]']] = interval_conf
            res2[['odds_2.5', 'odds_97.5']] = odds_r
                       
            bigdata = pd.concat([res,res2], axis = 1)                
                      
            sintese += bigdata
            
            start_2 = start_2 + init
            start_3 = start_3 + init 
      
(time.time() - start)

var_base_simu.write(var_base_simu, file = "E:/Rose/Standard_menard/Simulation/oversampling/smote5/base_simula.csv", sep = "|",dec = ".",row.names = FALSE)
sintese.write(sintese, file = "E:/Rose/Standard_menard/Simulation/oversampling/smote5/base_final.csv" ,sep = "|",dec = ".",row.names = FALSE)
   