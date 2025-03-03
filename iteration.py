# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:27:01 2025

@author: 304723
"""
#%%

import pandas as pd
from main_function import main_function
import matplotlib.pyplot as plt
import numpy as np


old_ratio = None

#%% 
step_size = 10
mape = 0  #.21 is base case


ratios = list(range(0,100+step_size,step_size))
#ratios = [30]*10
#ratios = [50,70]
costs = pd.DataFrame({'CBC costs': len(ratios)*0.0,
                       'RD costs': len(ratios)*0.0,
                       'total_costs': len(ratios)*0.0,
                       'market_costs': len(ratios)*0.0,
                       'market_price': len(ratios)*0.0},
                         index = ratios)  
index = 0
for i in ratios:
    print(f'tested ratio = {i}, itteration {index}\n')
    results = main_function(i/100,index,old_ratio,mape)
    costs_tuple = results[0:5]
    old_ratio = results[5]
    costs.iloc[index] = costs_tuple
    index+=1  
 
for index,row in costs.iterrows():
    if sum(row) == 0:
        costs= costs.drop(index)
 
    
 
#%%
fig, ax = plt.subplots()
pd.DataFrame(costs.iloc[:, :3]).plot(ax=ax)  # Remove space after pd.
ax.set_xlabel("% of congestion solved by CBC")
ax.set_xticks(np.arange(0, 110, step=10))
ax.set_ylabel("Operational Costs [Euro] ")
ax.set_title("Costs VS CBC/RD ratio (CHP sell orders = 30")
plt.show()


fig, ax = plt.subplots()
pd.DataFrame(costs.loc[:,'market_price']).plot(ax=ax)
ax.set_xlabel("% of congestion solved by CBC")
ax.set_xticks(np.arange(0, 110, step=10))
ax.set_ylabel("Total market Costs [K Euro]")
ax.set_title("Market price VS CBC/RD ratio (CHP sell orders = 30)")
plt.show()


plt.boxplot(costs['total_costs']/costs['total_costs'].mean())
plt.show()