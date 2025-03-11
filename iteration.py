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
df_dp = pd.DataFrame({'Change in renewable generation' : len(ratios)*0.0, 
                      'Change in conventional generation' : len(ratios)*0.0,
                      'Change in consumption' : len(ratios)*0.0}, 
                     index = ratios)
index = 0
for i in ratios:
    print(f'tested ratio = {i}, itteration {index}\n')
    results = main_function(i/100,index,old_ratio,mape)
    costs_tuple = results[0:5]
    old_ratio = results[5]
    
    #part that summarizes the Delta power data
    CBC_orderbook = results[6]
    RD_orderbook = results[7]
    dp_CBC = results[8]
    dp_RD = results[9]
    
    delta_RE          = 0
    delta_CHP         = 0
    delta_consumption = 0
    
    if dp_CBC != 0:
        
        for (asset, time), value in dp_CBC.items():  # Correctly iterate over dictionary keys and values
            if value != 0.0:
                RE_mask = (CBC_orderbook['asset'] == asset) & (CBC_orderbook['delivery_start'] == time) & (CBC_orderbook['type'] == 'RE')
                CHP_mask = (CBC_orderbook['asset'] == asset) & (CBC_orderbook['delivery_start'] == time) & (CBC_orderbook['type'] == 'CHP')
                industry_mask = (CBC_orderbook['asset'] == asset) & (CBC_orderbook['delivery_start'] == time) & (CBC_orderbook['type'] == 'Industry')
                if RE_mask.any():  # Check if any row matches
                    delta_RE -= value
                    delta_CHP += value
                if CHP_mask.any():
                    delta_CHP -= value
                if industry_mask.any():
                    delta_consumption += value
        
        for (asset, time), value in dp_RD.items():  # Correctly iterate over dictionary keys and values
            if value != 0.0:
                RE_mask = (RD_orderbook['asset'] == asset) & (RD_orderbook['delivery_start'] == time) & (RD_orderbook['type'] == 'RE')
                CHP_mask = (RD_orderbook['asset'] == asset) & (RD_orderbook['delivery_start'] == time) & (RD_orderbook['type'] == 'CHP')
                industry_mask = (RD_orderbook['asset'] == asset) & (RD_orderbook['delivery_start'] == time) & (RD_orderbook['type'] == 'Industry')
                if RE_mask.any():  # Check if any row matches
                    delta_RE -= value
                if CHP_mask.any():
                    delta_CHP -= value
                if industry_mask.any():
                    delta_consumption += value

    df_dp.iloc[index, 0] = delta_RE
    df_dp.iloc[index, 1] = delta_CHP
    df_dp.iloc[index,2] = delta_consumption
    
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
ax.set_ylabel("Whole-sale price [Euro/MW]")
ax.set_title("Market price VS CBC/RD ratio (CHP sell orders = 30)")
plt.show()

fig, ax = plt.subplots()
df_dp.loc[~(df_dp.iloc[:, :3] == 0).all(axis=1), :].plot(ax=ax)
ax.set_xlabel("% of congestion solved by CBC")
ax.set_xticks(np.arange(0, 110, step=10))
ax.set_ylabel("Change in production/generation (GW)")
ax.set_title("Impact of CBC/RD on energy generation and consumption \n (+ is more production and consumption)")
plt.show()

plt.boxplot(costs['total_costs']/costs['total_costs'].mean())
plt.show()