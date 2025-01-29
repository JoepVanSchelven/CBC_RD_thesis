# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:27:01 2025

@author: 304723
"""
import pandas as pd
from main_function import main_function
import matplotlib.pyplot as plt

ratios = range(0,100,10)
costs = pd.DataFrame({'CBC costs': len(ratios)*0.0,'RD costs': len(ratios)*0.0,'total_costs': len(ratios)*0.0},index = ratios)  
    
for i in ratios:
    print(f'tested ratio = {i}\n')
    costs_tuple = main_function(i/100)
    costs.loc[i] = costs_tuple
    
fig, ax = plt.subplots()
pd. DataFrame(costs).plot(ax=ax)
ax.set_xlabel("% of congestion solved by CBC")
ax.set_ylabel("Costs [Euro]")
ax.set_title("Costs VS CBC/RD ratio")
