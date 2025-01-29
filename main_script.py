#!/usr/bin/env python
# coding: utf-8
# %%

# #  CBC/RD activation model
# Joep van Schelven - 2025 
# 
# This model is aimed to simulate different activation strategies of Capacity limitng contracts (CLC/CBC) and redispatch contracts. The main goal is to effectively mitigate congestion against optimal costs.
# 

# ### import packages

# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os  

#this is a custom script used to plot different networks
from network_plotting import draw_network, draw_network_with_power_flows, draw_network_with_absolute_power_flows, draw_network_with_congestion



# ### Load variables and input data
# The input data needs to be a seperate Excel with specified lay-out
# 
# note: generation is negative and load is positive

# %%


from config_loader import retrieve_config_variables

# Retrieve variables as a tuple, use this line in sperate files to get teh same gloabl variables
ptus, input_file, susceptance, df_lines, buses, n_buses, n_lines, ratio = retrieve_config_variables()

df_loads_D2 = pd.read_excel(input_file,'loads', header=0, index_col=0)  #load D-2 prognoses for loads
df_RE_D2 = pd.read_excel(input_file,'re', header=0, index_col=0)        #Load D-2 prognoses for renewable generation
df_chp_max = pd.read_excel(input_file,'chp_max', header=0, index_col=0) #Load the maximum power output of CHPs


# ### DC load-flow
# First, we will use the the D-2 prognoses to perform a 'manual' DC load-flow. This load-flow will be used to visualise the network behaviour nd see where the congestion occurs. 

# %%


#Make a matrix where every node and every ptu have total load excluding CHP generation
load_per_node_D2 = np.zeros((n_buses,ptus))

# This is afunction to add the values of one DF to another array, based on the nodes provided in the DF
def add_to_array(df_in: pd.DataFrame, np_out: np.array) -> np.array:
    for idx, row in df_in.iterrows():  # Iterate over DataFrame rows
        node = row["node"]  # Use the "node" column explicitly
        if node in buses:  # Check if the node is in the buses list
            bus_idx = buses.index(node)  # Get the index of the node in the buses list
            values = row[1:].to_numpy(dtype=float)  # Convert the remaining columns to a NumPy array
            np_out[bus_idx, :] += values  # Add values to the corresponding row in np_out
    return np_out

# use the function to add RE and load profiels to the load per node
load_per_node_D2 = add_to_array(df_RE_D2, load_per_node_D2)
load_per_node_D2 = add_to_array(df_loads_D2, load_per_node_D2)

#This is a function that identifies the imbalnce at every PTU and dispatches the CHPs to balance the system
    # The CHPs are dispatched in order, to mimic a merit order (highset = cheapest)
    
def CHP_dispatch_calc(df_chp_max: pd.DataFrame, load_per_node:pd.DataFrame) -> pd.DataFrame:
    # Initialize an array for CHP dispatch
    chp_dispatch = np.zeros((len(df_chp_max), ptus))
    #find total imbalance per ptu this is the power the chp_maxS have to deliver for a balanced grid
    imbalance_per_ptu = load_per_node.sum(axis=0)
    # Calculate power required per ptu
    p_chp_required = -imbalance_per_ptu
    
    # Loop over each PTU
    for t in range(ptus):
        p = p_chp_required[t]
        chp = 0
    
        # Balance the power for this PTU
        while p < 0 and chp < len(df_chp_max):  # Ensure we don't exceed available CHPs
            max_dispatch = df_chp_max.iloc[chp, 1]  # Maximum dispatch capacity for this CHP
            
            if abs(p) <= abs(max_dispatch):  # If the current CHP can fully balance the power
                chp_dispatch[chp, t] = p
                p = 0  # Fully balanced
            else:  # Dispatch as much as possible from the current CHP
                chp_dispatch[chp, t] = max_dispatch
                p -= max_dispatch  # Remaining imbalance
                chp += 1  # Move to the next CHP
    
        p_chp_required[t] = p  # Update remaining imbalance for this PTU (should not become positive)
    
    #make a DF where the location of the CHPs is also included
    chp = pd.DataFrame(chp_dispatch)
    chp.insert(0, 'node', df_chp_max.iloc[:,0])
    return chp

#use both functions to add the CHP to the load per node.
chp = CHP_dispatch_calc(df_chp_max, load_per_node_D2)
load_per_node_D2 = add_to_array(chp, load_per_node_D2)

# check if the system is balanced, exit the script if not the case
if load_per_node_D2.sum() != 0:
    fig, ax = plt.subplots()
    pd. DataFrame(load_per_node_D2.sum(axis = 0)).plot(ax=ax)
    ax.set_xlabel("Hour on day D")
    ax.set_ylabel("Unbalance [MW]")
    ax.set_title("Unbalance per hour ")
    if load_per_node_D2.sum() > 0:
        sys.exit(f' The system is inherently unbalanced because too little generation. {load_per_node_D2.sum()} MWh additional generation is required is required.\n')
    elif load_per_node_D2.sum < 0:
        sys.exit(f' The system is inherently unbalanced because to much RE. {load_per_node_D2.sum()} MWh "curtailement" is required.\n')
    else:
        print('D-2 prognosis is balanced, load flow calculation will be performed\n')
#build the susceptance matrix
df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance))
B = np.zeros((n_buses, n_buses))

for _, row in df_lines.iterrows():
    from_idx = buses.index(row['from_bus'])
    to_idx = buses.index(row['to_bus'])
    B[from_idx, to_idx] -= row['susceptance']
    B[to_idx, from_idx] -= row['susceptance']
    B[from_idx, from_idx] += row['susceptance']
    B[to_idx, to_idx] += row['susceptance']
    
    
#define a function to calculate the powerflows according to DC powerflow approach
def calculate_powerflow(df_loads_D2: np.array) -> pd.DataFrame:
    # Initialize results df
    df_results = pd.DataFrame({'flow': n_lines * [0.0]}, index=df_lines.index)

    slack_bus = 0 # note: because system is always balanced a slackbus is not nescsary and it does nt matter wich bus you define
    B_reduced = np.delete(B, slack_bus, axis=0)
    B_reduced = np.delete(B_reduced, slack_bus, axis=1)

    # Adjust the load vector to exclude the slack bus
    load_reduced = np.delete(df_loads_D2, slack_bus, 0)

    # Solve for the phase angles (theta) at the non-slack buses
    theta_reduced = np.linalg.solve(B_reduced, load_reduced)

    # Reconstruct the full theta vector including the slack bus (assumed 0 phase angle)
    theta = np.zeros(n_buses)
    theta[1:] = theta_reduced

    # Calculate line flows    
    df_results['flow'] = [
        row['susceptance'] * (theta[buses.index(row['from_bus'])] - theta[buses.index(row['to_bus'])])
        if row['from_bus'] in buses and row['to_bus'] in buses else 0
        for _, row in df_lines.iterrows()
    ]
    return df_results

#use function to create a DF with all the flows per line 

df_flows_D2 = pd.DataFrame(columns = range(ptus))
for t in range(ptus):
    df_flows_D2[t] = calculate_powerflow(load_per_node_D2[:,t])

#draw your network
draw_network(df_lines)


# ### Localise and deterimine congestion volume
# Use the flow DF and the capacity of the lines to find the congestion

# %%


# initieer een DF voor alle overloads
df_congestion_D2 = pd.DataFrame(index=df_lines.index, columns = range(ptus))

# definieer een functie die de overload kan bereken aan de hand van line capacity en flows
def overload_calculation(df_flows : pd.DataFrame) -> pd.DataFrame:
    for line in df_congestion_D2.index:  # Iterate over line indices
        for t in range(ptus):  # Iterate over PTUs (time steps)
            overload = abs(df_flows.iloc[line,t]) - df_lines['capacity'].loc[line]
            if overload > 0:
                df_congestion_D2.iloc[line, t] = float(overload)
            else:
                df_congestion_D2.iloc[line, t] = 0
    return df_congestion_D2

df_congestion_D2 = overload_calculation(df_flows_D2)
congestion_D2 = sum(df_congestion_D2.sum())


# %%


# %% Vsualization of generation per type and per node of the d-2 prognoses

#make a dict with all the load per generation type
load_per_type = {'df_loads_D2': ptus * [0], 'df_RE_D2': ptus * [0], 'chp': ptus * [0]}
for key in load_per_type.keys():
    load_per_type[key] = globals()[key].iloc[:,1:].sum()
    
load_per_type = pd.DataFrame(load_per_type)
load_per_type['imbalance'] = load_per_type.sum(axis=1)


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot 1: df_loads_D2 per type before RD and CLC
load_per_type.plot(ax=axes[0, 0])
axes[0, 0].set_xlabel("Hour on day D")
axes[0, 0].set_ylabel("Active Power [MW]")
axes[0, 0].set_title("df_loads_D2 per type before RD and CLC")

# Plot 2: df_loads_D2 per node before RD and CLC
pd.DataFrame(load_per_node_D2).T.plot(ax=axes[0, 1])
axes[0, 1].set_xlabel("Hour on day D")
axes[0, 1].set_ylabel("Active Power [MW]")
axes[0, 1].set_title("df_loads_D2 per node before RD and CLC")

# Plot 3: absolute flow per line before RD and CLC
pd.DataFrame(abs(df_flows_D2)).T.plot(ax=axes[1, 0])
axes[1, 0].set_xlabel("Hour on day D")
axes[1, 0].set_ylabel("Absolute Flow [MW]")
axes[1, 0].set_title("Absolute Flow per line before RD and CLC")

# Plot 4: Congestion per line before RD and CLC
pd.DataFrame(df_congestion_D2).T.plot(ax=axes[1, 1])
axes[1, 1].set_xlabel("Hour on day D")
axes[1, 1].set_ylabel("Congestion [MW]")
axes[1, 1].set_title(f"Congestion per line before RD and CLC (total: {round(congestion_D2, 2)})")

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

# Create 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot 1: Network with total congestion
congestion_per_bus = df_congestion_D2.sum(axis='columns')
congestion_per_bus = congestion_per_bus.rename_axis('line').reset_index(name='flow')
draw_network_with_power_flows(df_lines, congestion_per_bus, 'Network with total congestion', axes[0, 0])

# Plot 2: Network with peak congestion
congestion_peak_per_bus = df_congestion_D2.max(axis='columns')
congestion_peak_per_bus = congestion_peak_per_bus.rename_axis('line').reset_index(name='flow')
draw_network_with_power_flows(df_lines, congestion_peak_per_bus, 'Network with peak congestion', axes[0, 1])

# Plot 3: Congestion per hour
df_congestion_D2[df_congestion_D2.sum(axis=1) != 0].transpose().plot(ax=axes[1, 0])
axes[1, 0].set_xlabel("Hour on day D")
axes[1, 0].set_ylabel("Congestion [MW]")
axes[1, 0].set_title("Congestion per hour")

# Leave the last subplot empty
axes[1, 1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

#check if there is congstion in the system
if congestion_D2 == 0:
    sys.exit('No congestion is the system\n')


# ### Activate CBCs
# Het Idee is dat eerst CBCs worden afgeroepen (optimalisereend voor de kosten en het congestievolume terugdringend naar een bepaald niveau).
# Vervolgens word er weer gedispatched met als constraint dat he tcongestievolume niet mag toenemen

# %%
df_CBC_orderbook = pd.read_excel(input_file,'CBC', header=0, index_col=0) #read the orderbook

#add all activated RE and CHP to the CBC orderbook, in order to do this, an assumption about the D-1 price is made
prognosis_wholesale_price = df_chp_max.loc[len(chp[chp.iloc[:, 1:].sum(axis=1) < 0])-1,'price']
CBC_RE_premium = 0.1*prognosis_wholesale_price #compensation on top off the wholesale market prognosis for RE
CBC_CHP_premium = 10*prognosis_wholesale_price #compensation on top off the wholesale market prognosis for CHPs

def add_generation_to_orderbook(generation:pd.DataFrame,orderbook:pd.DataFrame,premium:float = 0)->pd.DataFrame:
    for index, row in generation.iloc[:, 1:].iterrows(): #This loop adds a bid for all expected RE production
        for t, value in row.items():
            if value < 0:
                list_order = [[generation.iloc[index, 0], t, t+1, 'buy', prognosis_wholesale_price+premium, -value]]
                orderbook = pd.concat([orderbook, pd.DataFrame(list_order, columns=orderbook.columns)], ignore_index=True)
    return orderbook

df_CBC_orderbook = add_generation_to_orderbook(df_RE_D2, df_CBC_orderbook, CBC_RE_premium)
df_CBC_orderbook = add_generation_to_orderbook(chp, df_CBC_orderbook, CBC_CHP_premium)





(load_per_node, df_CBC_orderbook, congestion) = (load_per_node_D2, df_CBC_orderbook,sum(df_congestion_D2.sum()))
Tee = 0
ratio = ratio

df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance))
max_CLC_up = np.zeros((n_buses, ptus)) #the maximum increase in power (more consumption/less production) at a given location
max_CLC_down = np.zeros((n_buses, ptus)) #the maximum decrease in power (less consumption/more production) at a given location
CLC_price_up = np.zeros((n_buses, ptus))
CLC_price_down = np.zeros((n_buses, ptus))
p_prognosis = load_per_node.copy()



for _, row in df_CBC_orderbook.iterrows():
    new_power = row['power'] if row['buy/sell'] == 'buy' else - row['power']
    if new_power > 0:   # Create an array with the maximum upward per per time/octino and a dict with the corresponding price
        max_CLC_up[row['bus'], row['delivery_start']: row['delivery_end']] += new_power
        CLC_price_up[row['bus'], row['delivery_start']: row['delivery_end']] = row['price']

    elif new_power < 0:     # Create an array with the maximum downward per per time/octino and a dict with the corresponding price 
        max_CLC_down[row['bus'], row['delivery_start']: row['delivery_end']] += new_power
        CLC_price_down[row['bus'], row['delivery_start'] : row['delivery_end']] = row['price']

# Convert to dicts for pyomo initialization
max_CLC_up_dict = {index: value for index, value in np.ndenumerate(max_CLC_up)}
max_CLC_down_dict = {index: value for index, value in np.ndenumerate(max_CLC_down)}
CLC_price_up_dict = {index: value for index, value in np.ndenumerate(CLC_price_up)}
CLC_price_down_dict = {index: value for index, value in np.ndenumerate(CLC_price_down)}
p_prognosis_dict = {index: value for index, value in np.ndenumerate(p_prognosis)}
 

model = pyo.ConcreteModel() # build the model
# Define sets (indices)
model.bus_set = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
model.line_set = pyo.RangeSet(0, n_lines - 1)
model.time_set = pyo.RangeSet(0, ptus-1)  

# Define parameters
model.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
model.max_dp_up = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=max_CLC_up_dict)
model.price_up = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=CLC_price_up_dict)
model.max_dp_down = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=max_CLC_down_dict)
model.price_down = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=CLC_price_down_dict)
model.p_prognosis = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=p_prognosis_dict)

# Define variables
model.theta = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals) # angle for DCLF
model.f = pyo.Var(model.line_set, model.time_set, within=pyo.Reals) #flow
model.congestion = pyo.Var(model.line_set, model.time_set, within=pyo.NonNegativeReals)  # >= 0
model.p = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # power
model.dp = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # dp after CLC
model.balancing_p = pyo.Var(model.time_set, within=pyo.Reals)  # Power uesd for distributed slackbus
model.u = pyo.Var(model.bus_set, model.time_set, within=pyo.Binary)  # Binary variable for condition

model.total_congestion = pyo.Var(within=pyo.Reals) #sum of congestion
model.total_costs = pyo.Var(within=pyo.Reals)   #costs


# Constraints to link `u` and `dp`. u is binary (0/1)  the following expresisons determine that if u is 1, dp has to be any value between 0 an 1e6
    #if u==0, dp is a negative price. This allow the objective funciton to add negative and positve prices. and allos for an upward bid and downward bid per location
def link_u_dp_rule(m, b, t):
    return m.dp[b, t] <= m.u[b, t] * 1e9  # Arbitrarily large value             dp is kliener dan 0 (u=0) of 1e9 (u=1)
model.link_u_dp = pyo.Constraint(model.bus_set, model.time_set, rule=link_u_dp_rule)

def link_u_dp_neg_rule(m, b, t):
    return m.dp[b, t] >= -1e9 * (1 - m.u[b, t])  # Arbitrarily large value      dp is groter dan -1e9 (u=0) of 0 (u=1)
model.link_u_dp_neg = pyo.Constraint(model.bus_set, model.time_set, rule=link_u_dp_neg_rule)


# Define Constraints
def def_congestion_1(m, l, t): #Congestion in case flow is positive
    capacity = df_lines.iloc[l]['capacity']
    return m.congestion[l, t] >= m.f[l, t] - capacity 

model.con_def_congestion_1 = pyo.Constraint(model.line_set, model.time_set, rule=def_congestion_1)

def def_congestion_2(m, l, t):  #Congestion in case flow is negative
    capacity = df_lines.iloc[l]['capacity']
    return m.congestion[l, t] >= -m.f[l, t] - capacity 

model.con_def_congestion_2 = pyo.Constraint(model.line_set, model.time_set, rule=def_congestion_2)
'''
# Define the zero congestion constraint
def zero_congestion_constraint(m):
    return sum(m.congestion[l, t] for l in m.line_set for t in m.time_set) == 0

# Add the constraint to the model
model.zero_congestion = pyo.Constraint(rule=zero_congestion_constraint)
'''
# DC powerflow equations
def ref_theta(m, t):
    return m.theta[0, t] == 0.0

model.con_ref_theta = pyo.Constraint(model.time_set, rule=ref_theta)

def DC_flow(m, l, t):
    from_bus = df_lines.iloc[l]['from_bus'] 
    to_bus = df_lines.iloc[l]['to_bus'] 
    susceptances = df_lines.iloc[l]['susceptance']
    return m.f[l, t] == susceptances * (m.theta[from_bus, t] - m.theta[to_bus, t])

model.con_DC_flow = pyo.Constraint(model.line_set, model.time_set, rule=DC_flow)

def nodal_power_balance(m, b, t):
    inflows = sum(m.f[l, t] for l in model.line_set if df_lines.iloc[l]['to_bus'] == b)
    outflows = sum(m.f[l, t] for l in model.line_set if df_lines.iloc[l]['from_bus'] == b)
    return inflows - outflows + m.p[b, t] == 0

model.con_nodal_power_balance = pyo.Constraint(model.bus_set, model.time_set, rule=nodal_power_balance)

# Define balancin_p (distrubeted slack)
def def_balancing_p(m, b, t):
    total_imbalance = sum(m.dp[b, t] for b in model.bus_set)
    n_buses = len(m.bus_set)
    return m.balancing_p[t] == total_imbalance / n_buses

model.con_def_balancing_p = pyo.Constraint(model.bus_set, model.time_set, rule=def_balancing_p )
# Define p
def def_p(m, b, t):
    return m.p[b, t] == m.p_prognosis[b, t] + m.dp[b, t] - m.balancing_p[t]

model.con_def_p = pyo.Constraint(model.bus_set, model.time_set, rule=def_p)

# Bound dp from below and above
def lower_bound_dp(m, b, t):
    if m.max_dp_down[b, t] >= 0:
        return m.dp[b, t] >= 0
    else:
        return m.dp[b, t] >= m.max_dp_down[b, t]
    
model.con_lower_bound_dp = pyo.Constraint(model.bus_set, model.time_set, rule=lower_bound_dp)
    
def upper_bound_dp(m, b, t):
    if m.max_dp_up[b, t] >= 0:
        return m.dp[b, t] <= m.max_dp_up[b, t]
    else:
        return m.dp[b, t] <= 0.0

model.con_upper_bound_dp = pyo.Constraint(model.bus_set, model.time_set, rule=upper_bound_dp)


# Objective function
def total_costs_def(m):
    return m.total_costs == sum(
        m.dp[b, t] * (m.price_up[b, t] * m.u[b, t] + m.price_down[b, t] * (1 - m.u[b, t])) * m.dt
        for b in m.bus_set
        for t in m.time_set
    )

model.total_costs_constraint = pyo.Constraint(rule=total_costs_def)
# Add congestion defition
def def_total_congestion(m):
    return m.total_congestion ==  sum(sum(m.congestion[l, t] * m.dt for l in m.line_set) for t in m.time_set)

model.con_def_total_congestion = pyo.Constraint(rule=def_total_congestion)

#'''
#Objective function only costs
def decrease_congestion_constraint(m):
    return m.total_congestion == congestion * (1-ratio)

# Add the constraint to the model
model.decrease_congestion = pyo.Constraint(rule=decrease_congestion_constraint)

def objective_function(m):
    return m.total_costs

#'''


'''
#Objective function, congestion and costs mixxed
def objective_function(m):
   return (1.0 - epsilon) * m.total_congestion + epsilon * m.total_costs
'''



model.objective_function = pyo.Objective(sense=pyo.minimize, expr=objective_function)


# Store model (convenient for debugging)
path = os.getcwd() + '\\'
model.write(path + "model_CBC.lp", io_options={"symbolic_solver_labels": True})

#%% Solve the model
opt = SolverFactory('gurobi')
opt.options['mipgap'] = 0.001
opt.options['DualReductions'] = 0
results = opt.solve(model, tee=Tee)

termination_condition = results.solver.termination_condition

if termination_condition == pyo.TerminationCondition.infeasible:
    sys.exit("Model is infeasible.\n")
elif termination_condition == pyo.TerminationCondition.unbounded:
    sys.exit("Model is unbounded.\n")
elif termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
    sys.exit("Model is either infeasible or unbounded.\n")
else:
    print(f"Termination condition: {termination_condition}\n")

total_congestion_results = pyo.value(model.total_congestion)
total_costs_results = pyo.value(model.total_costs)    
print(total_costs_results)
#print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)} = {total_congestion_results * (np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)}\n")
print(f"total costs are {total_costs_results}\n")    
print(ratio)

sys.exit('stop after CBC')
"""
from RD_CBC_functions import optimal_CBC

model = optimal_CBC(load_per_node_D2, df_CBC_orderbook,sum(df_congestion_D2.sum()),0)

# Get results from the model
p_CBC = {(b, t): pyo.value(model.dp[b, t]) for b in model.bus_set for t in model.time_set}
congestion_hypotheses_CBC = {(l, t): pyo.value(model.congestion[l, t]) for l in model.line_set for t in model.time_set}

total_congestion_results = pyo.value(model.total_congestion)
total_costs_CBC = pyo.value(model.total_costs)


# Create a DataFrame from the dictionary
df_dp_CBC = pd.DataFrame.from_dict(p_CBC, orient='index', columns=['value'])
# Reset the index and expand the tuple keys into separate columns
df_dp_CBC.index = pd.MultiIndex.from_tuples(df_dp_CBC.index, names=["node", "hour"])
df_dp_CBC = df_dp_CBC.unstack(level="hour")

# Clean up the column names
df_dp_CBC.columns = df_dp_CBC.columns.droplevel(0)
df_dp_CBC = df_dp_CBC.reset_index()


# Transform the dictionary into a DataFrame
df_congestion_hypotheses_CBC = pd.DataFrame.from_dict(congestion_hypotheses_CBC, orient="index", columns=["congestion"])
# Reset the index and expand the tuple keys into separate columns
df_congestion_hypotheses_CBC.index = pd.MultiIndex.from_tuples(df_congestion_hypotheses_CBC.index, names=["node", "hour"])
df_congestion_hypotheses_CBC = df_congestion_hypotheses_CBC.unstack(level="hour")
# Clean up the column names
df_congestion_hypotheses_CBC.columns = df_congestion_hypotheses_CBC.columns.droplevel(0)



#add CBC result to load array
load_per_node_D1 = add_to_array(df_dp_CBC, load_per_node_D2.copy())

# Perform load to to see how the congestion is after CBC acivation

df_flows_D1 = pd.DataFrame(columns = range(ptus))
for t in range(ptus):
    df_flows_D1[t] = calculate_powerflow(load_per_node_D1[:,t])
    
df_congestion_D1 = overload_calculation(df_flows_D1)
congestion_D1 = sum(df_congestion_D1.sum())

tot_congestion_hypotheses_CBC = sum(df_congestion_hypotheses_CBC.sum()) # how much congestion was expected to be left after distributed slack CBC activtion
ratio_actual =  1- (congestion_D1 / congestion_D2)
print(f'The aimed congestion reduction is {1-(tot_congestion_hypotheses_CBC/congestion_D2)}, the actual congestion redution is {ratio_actual}\n')


# ### Actualise prognoses
# Introduce 'noise' to the prognoses so they represent the actual T-pofile data to be used for the marketcoupling later the CBC will also be taken into consoderation during this stage, but not yet implemented

# %%


def add_normal_noise(df_D2 : pd.DataFrame, std : int) -> pd.DataFrame: #function to add normal noise to with provided STD DF with specific shape (mean=0) Only adds noise to non-zero values
    df_output = df_D2.copy()
    for load in range(len(df_D2)):
        noise = np.random.normal(0, std, (df_D2.columns != 'node').sum())
        noise = np.trunc(noise * 10**2) / 10**2
        df_output.iloc[load,1:] += (df_output.iloc[load, 1:] != 0) * noise
    return df_output

# add noise to the D-2 to make 'actual' data. use these to make new load per node, and new CHP dispatch (marketcoupling). Results in a new load per node and new congestion 
df_loads = add_normal_noise(df_loads_D2, 0)
df_RE    = add_normal_noise(df_RE_D2, 0)

load_per_node = np.zeros((n_buses,ptus))
load_per_node = add_to_array(df_RE, load_per_node)
load_per_node = add_to_array(df_loads, load_per_node)
load_per_node = add_to_array(df_dp_CBC, load_per_node)


chp_coupling = CHP_dispatch_calc(df_chp_max, load_per_node)
#find total load per node per CHP 

load_per_node = add_to_array(chp_coupling, load_per_node)

"""
# %%


# %% Vsualization of generation per type and per node before RD

#make a dict with all the load per generation type
load_per_type = {'df_loads': ptus * [0], 'df_RE': ptus * [0], 'chp_coupling': ptus * [0], 'df_dp_CBC': ptus * [0]}
for key in load_per_type.keys():
    load_per_type[key] = globals()[key].iloc[:,1:].sum()
    
load_per_type = pd.DataFrame(load_per_type)
load_per_type['imbalance'] = load_per_type.sum(axis=1)


fig, ax = plt.subplots()
load_per_type.plot(ax=ax)
ax.set_xlabel("Hour on day D")
ax.set_ylabel("Active Power [MW]")
ax.set_title("df_loads per type before RD ")

fig, ax = plt.subplots()
pd.DataFrame(load_per_node).T.plot(ax=ax)
ax.set_xlabel("Hour on day D")
ax.set_ylabel("Active Power [MW]")
ax.set_title("df_loads per node before RD")

# Truncate load_per_node after the th decimal
scaling_factor = 10**6
load_per_node = np.trunc(load_per_node * scaling_factor) / scaling_factor
# check if the system is balanced, exit the script if not the case
if load_per_node.sum() != 0:
    fig, ax = plt.subplots()
    pd. DataFrame(load_per_node_D2.sum(axis = 0)).plot(ax=ax)
    ax.set_xlabel("Hour on day D")
    ax.set_ylabel("Unbalance [MW]")
    ax.set_title("Unbalance per hour ")
    if load_per_node.sum() > 0:
        sys.exit(f' The system is inherently unbalanced because too little generation after CBC activation {load_per_node.sum()} MWh additional generation is required is required.\n')
    elif load_per_node.sum() < 0:
        sys.exit(f' The system is inherently unbalanced because to much RE after CBC activation. {load_per_node.sum()} MWh "curtailement" is required.\n')
else:
    print('Balanced market coupling succesful, load flow calculation will be performed\n')
sys.exit('Stop before cbc')
# ### Redispatch
# Using Pyomo, and load-flow constraints, dispatch the optimal set of bids to minimze costs and mitigate any remaining congestion. 
# Input shouldbe a balanced DF wih load per node per ptu

# %%
#sys.exit('stop before RD\n')

#read orderbook
df_RD_orderbook = pd.read_excel(input_file,'RD', header=0, index_col=0) #read the orderbook

from RD_CBC_functions import optimal_redispatch

model = optimal_redispatch(load_per_node, df_RD_orderbook)


# %%


# Get results from the model
dp_results = {(b, t): pyo.value(model.dp[b, t]) for b in model.bus_set for t in model.time_set}
f_results = {(l, t): pyo.value(model.f[l, t]) for l in model.line_set for t in model.time_set}
congestion_results = {(l, t): pyo.value(model.congestion[l, t]) for l in model.line_set for t in model.time_set}

total_congestion_results = pyo.value(model.total_congestion)
total_costs_RD = pyo.value(model.total_costs)


total_costs = total_costs_RD + total_costs_CBC

# Create a DataFrame from the dictionary
result_df_dp = pd.DataFrame.from_dict(dp_results, orient='index', columns=['value'])
result_df_f = pd.DataFrame.from_dict(f_results, orient='index', columns=['value'])
result_df_congestion = pd.DataFrame.from_dict(congestion_results, orient='index', columns=['value'])


# %%




