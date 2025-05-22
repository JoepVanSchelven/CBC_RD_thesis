# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:27:01 2025

@author: 304723
"""
from time import monotonic
start_time = monotonic()
import pandas as pd
from alternative_function import alt_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
plt.rcParams['font.family'] = 'Times New Roman'
#%% User defined parameters
simulations_per_ratio = 1664 #1664 for 98% CI
noise = 0.21
security_of_supply = .99
safety_margins = [0,.5,1.0,1.1,1.15,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.5,2.0]

#%%
expected_time = 4.1*simulations_per_ratio*len(safety_margins)

print(f'Expected time of entrie simulation: {expected_time/60:.2f} minutes')

old_ratio = None
i = 0
#%%


#Initialize DataFrames
costs = pd.DataFrame(
    np.zeros((len(safety_margins), 5)),  # (rows, columns)
    columns=['CBC costs', 'RD costs', 'total_costs', 'market_costs', 'market_price'],
    index = safety_margins)
simulations_per_margin = list(range(simulations_per_ratio))

monte_carlo_costs = pd.DataFrame(np.zeros((len(simulations_per_margin), 5)), 
                                  columns=['CBC costs', 'RD costs', 'total_costs', 'market_costs', 'market_price'], 
                                  index=simulations_per_margin)
#can be use to check if there is engough conversion
all_results = {}
old_margin = None

for index, margin in enumerate(safety_margins):
    print(f"Running simulations for safety margin {margin}")
    infeasible_count = 0
    #feasible = True  # Assume feasibility

    for MC in simulations_per_margin:
        #print(f"  Monte-Carlo iteration {MC}")
        print(f' \n Itteration {MC}')
        results = alt_function(margin, i, old_margin, noise)  # Run main function
        costs_tuple = results[0:5]
        old_margin = results[-1]
        i+=1
        monte_carlo_costs.iloc[MC] = costs_tuple  
        all_results[margin,MC] = costs_tuple[2]
        print(sum(costs_tuple))
        if sum(costs_tuple) == 0:  # Check for infeasibility
            print(f"  Infeasible solution found at margin {margin}")
            infeasible_count += 1   
            print(infeasible_count)
            print(infeasible_count/simulations_per_ratio > (1-security_of_supply))
            if infeasible_count/simulations_per_ratio > (1-security_of_supply):
                print(f'{margin} defenitely infeasible continue\n')
                break
           #break  # Stop further MC iterations for this ratio
    
    #check if more then a specified amount of solutinos was infeasible
    
    secure = True if infeasible_count/simulations_per_ratio <= (1-security_of_supply) else False 
    # If infeasible, store [0,0,0,0,0]; otherwise, compute mean
    result = monte_carlo_costs.mean() if secure else pd.Series([0, 0, 0, 0, 0], index=costs.columns)
    costs.iloc[index] = result  # Store result
    # Making a hist to chack normality
    # Extract values for the given ratio
    values = [all_results[key] for key in all_results if key[0] == margin]
    
    # Convert to a Pandas Series for plotting
    values_series = pd.Series(values)
    
    # Plot histogram
    values_series.hist()
    
    # Add a title
    plt.title(f"Histogram for ratio {margin}")
    plt.xlabel('Total operational costs')
    plt.show()
# Drop infeasible results
#costs = costs.loc[~(costs.sum(axis=1) == 0)]

#%% Plotting
 
def plot_data(data, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(dpi=1200)

    # Remove rows where all values are zero
    filtered_data = data.loc[~(data == 0).all(axis=1)]

    if len(filtered_data) == 1:  # If there's only one row
        margin = filtered_data.index[0]
        values = filtered_data.iloc[0]  # Extract the row as a Series

        # Create a DataFrame with slight index variations
        modified_data = pd.DataFrame(
            [values, values, values],  # Keep the values the same
            index=[margin - 0.02, margin, margin + 0.02]  # Slightly shift the index
        )

        modified_data.plot(ax=ax, linestyle='-')
    else:
        filtered_data.plot(ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(0, max(filtered_data.index), num=10))  # Generate 10 tick positions
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))  # Format to 2 decimals
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Calculate the minimal value of the x-axis data
    min_x_value = filtered_data.index.min()

    # Create a custom legend entry for the minimal x-value
    custom_legend = [Line2D([0], [0], color='w', lw=0, label=f'Minimal safety margin: {min_x_value:.2f}')]

    # Get the existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    if data.shape[1] == 1:
        # If there's only one line, add the custom legend entry without a line
        legend = ax.legend(handles=custom_legend, loc='upper left', frameon=True, handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)
    else:
        # If there are multiple lines, add the custom legend entry along with the existing handles
        legend = ax.legend(handles=handles + custom_legend, loc='upper left', frameon=True)

    plt.show()


# Plot operational costs
plot_data(
    pd.DataFrame(costs.iloc[:, :3]),  # Assuming mean across columns if multiple exist
    r"Safety margin",
    r"Operational Costs [k€]",
    #r"OPEX VS safety_margin"
)

# Plot market costs
plot_data(
    pd.DataFrame(costs.loc[:, 'market_price']),
    r"Safety margin",
    r"Market price [€/MW]",
    #r"D-1 market price VS safety margin ratio"
)

print(f"Expected duration = {expected_time} \n Actual duration = {monotonic() - start_time:.2f} seconds \n in minutes {(monotonic() - start_time)/60:.2f}")
