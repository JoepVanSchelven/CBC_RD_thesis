# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:27:01 2025

@author: 304723
"""
from time import monotonic
start_time = monotonic()
import pandas as pd
from main_function import main_function
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
#%% User defined parameters
simulations_per_ratio = 1664 #1664 for 98% CI
step_size_ratios = 5
noise = 0.21
security_of_supply = .999

#%%
expected_time = 3.6*((100/step_size_ratios)+1)*simulations_per_ratio

print(f'Expected time of entrie simulation: {expected_time/60:.2f} minutes')

old_ratio = None
i = 0
#%%
simulations_per_value = list(range(simulations_per_ratio))
ratios = list(range(0, 100+step_size_ratios , step_size_ratios))  # Ratio from 0% to 100%

#Initialize DataFrames
costs = pd.DataFrame(np.zeros((len(ratios), 5)), 
                      columns=['CBC costs', 'RD costs', 'total_costs', 'market_costs', 'market_price'], 
                      index=ratios)

monte_carlo_costs = pd.DataFrame(np.zeros((len(simulations_per_value), 5)), 
                                  columns=['CBC costs', 'RD costs', 'total_costs', 'market_costs', 'market_price'], 
                                  index=simulations_per_value)
#can be use to check if there is enough conversion
all_results = {}
for index, ratio in enumerate(ratios):
    print(f"Running simulations for ratio {ratio}")
    infeasible_count = 0
    #feasible = True  # Assume feasibility

    for MC in simulations_per_value:
        print(f"  Monte-Carlo iteration {MC}")

        results = main_function(ratio / 100, i, old_ratio, noise)  # Run main function
        costs_tuple = results[0:5]
        i+=1
        old_ratio = results[5]
        monte_carlo_costs.iloc[MC] = costs_tuple  
        all_results[ratio,MC] = costs_tuple[2]
        
        if sum(costs_tuple) == 0:  # Check for infeasibility
            print(f"  Infeasible solution found at ratio {ratio}")
            infeasible_count += 1
            if infeasible_count/simulations_per_ratio > (1-security_of_supply):
                print(f'{ratio} defentiely infeasible continue\n')
                break
            #feasible = False
           #break  # Stop further MC iterations for this ratio
    
    #check if more then a specified amount of solutinos was infeasible
    
    secure = True if infeasible_count/simulations_per_ratio <= (1-security_of_supply) else False 
    # If infeasible, store [0,0,0,0,0]; otherwise, compute mean
    result = monte_carlo_costs.mean() if secure else pd.Series([0, 0, 0, 0, 0], index=costs.columns)
    costs.iloc[index] = result  # Store result
    # Making a hist to chack normality
    # Extract values for the given ratio
    values = [all_results[key] for key in all_results if key[0] == ratio]
    
    # Convert to a Pandas Series for plotting
    values_series = pd.Series(values)
    
    # Plot histogram
    values_series.hist()
    
    # Add a title
    plt.title(f"Histogram for ratio {ratio}")
    plt.show()
# Drop infeasible results
#costs = costs.loc[~(costs.sum(axis=1) == 0)]

#%% Plotting
 
def plot_data(data, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(dpi=1200)
    
# Remove rows where all values are zero
    filtered_data = data.loc[~(data == 0).all(axis=1)]
    
        
    if len(filtered_data)==1:  # If there's only one row
            ratio = filtered_data.index[0]
            value = filtered_data.iloc[0]

            # Create slight variations
            modified_data = pd.Series(
                [value, value, value], 
                index=[ratio - 0.25*step_size_ratios , ratio, ratio + 0.25*step_size_ratios]
            )

            modified_data.plot(ax=ax, linestyle='-')
    else:
            filtered_data.plot(ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(0, 110, step=10))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


# Plot operational costs
plot_data(
    costs.iloc[:, :3],  # Assuming mean across columns if multiple exist
    r"$\vartheta$ [%]",
    r"Operational Costs [k€]",
    #r"Costs VS CBC/RD ratio"
)

# Plot market costs
plot_data(
    pd.DataFrame(costs.loc[:, 'market_price']),
    r"$\vartheta$ [%]",
    r"Total Market Price [k€]",
    #r"Market Costs VS CBC/RD ratio"
)

print(f"Expected duration = {expected_time} \n Actual duration = {monotonic() - start_time:.2f} seconds \n in minutes {(monotonic() - start_time)/60:.2f}")
