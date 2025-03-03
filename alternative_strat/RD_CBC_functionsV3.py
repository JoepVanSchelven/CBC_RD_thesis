# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:12:41 2025

@author: 304723
"""
from config_loader import retrieve_config_variables
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import os

    # Retrieve variables as a tuple, use this line in sperate files to get teh same gloabl variables
ptus, input_file, susceptance, df_lines, buses, n_buses, n_lines, ratio = retrieve_config_variables()
#%% RD prognosis function (some congetion alloed afserwards)
def optimal_redispatch_congestion_cost(load_per_node:pd.DataFrame, df_RD_orderbook:pd.DataFrame, Tee :int = 0) -> pyo.ConcreteModel:
    assets = list(np.unique(df_RD_orderbook.iloc[:,0].values))
    df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance)) #not a global parameter, so also have to do this function locally
    # Create a dictionary to map assets to their bus

    p_prognosis = load_per_node.copy()
    
    max_RD_up_dict = {}
    max_RD_down_dict = {}
    RD_price_up_dict = {}
    RD_price_down_dict = {}
    p_prognosis_dict = {index: value for index, value in np.ndenumerate(p_prognosis)}
    
    for asset in assets:  # Loop to construct the max_RD_up_dict
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'buy')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                max_power = filtered_df['power'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                max_RD_up_dict[key] = max_power
     
                        
    for asset in assets:  # Loop to construct the max_RD_down_dict
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'sell')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                max_power = -filtered_df['power'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                max_RD_down_dict[key] = max_power
    
    for asset in assets:  # Loop to construct the RD_price_up
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'buy')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                price_up = filtered_df['price'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                RD_price_up_dict[key] = price_up
                
                
    for asset in assets:  # Loop to construct the RD_price_down
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'sell')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                price_down = filtered_df['price'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                RD_price_down_dict[key] = price_down
         
    model_RD = pyo.ConcreteModel() # build the model_RD
   
    # Define sets (indices)
    model_RD.bus_set = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
    model_RD.line_set = pyo.RangeSet(0, n_lines - 1)
    model_RD.time_set = pyo.RangeSet(0, ptus - 1)  
    model_RD.asset_set = pyo.RangeSet(0, len(np.unique(df_RD_orderbook.iloc[:,0].values))-1)
    
    # Define parameters
    model_RD.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
    model_RD.max_dp_up = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=max_RD_up_dict)
    model_RD.price_up = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=RD_price_up_dict)
    model_RD.max_dp_down = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=max_RD_down_dict)
    model_RD.price_down = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=RD_price_down_dict)
    model_RD.p_prognosis = pyo.Param(model_RD.bus_set, model_RD.time_set, within=pyo.Reals, initialize=p_prognosis_dict)
    
    # Define variables
    model_RD.theta = pyo.Var(model_RD.bus_set, model_RD.time_set, within=pyo.Reals) # angle for DCLF
    model_RD.f = pyo.Var(model_RD.line_set, model_RD.time_set, within=pyo.Reals) #flow
    model_RD.congestion = pyo.Var(model_RD.line_set, model_RD.time_set, within=pyo.NonNegativeReals)  # >= 0
    model_RD.p = pyo.Var(model_RD.bus_set, model_RD.time_set, within=pyo.Reals)  # power
    model_RD.dp = pyo.Var(model_RD.asset_set, model_RD.time_set, within=pyo.Reals)  # dp after RD
    model_RD.u = pyo.Var(model_RD.asset_set, model_RD.time_set, within=pyo.Binary)  # Binary variable for condition
    
    
    model_RD.total_congestion = pyo.Var(within=pyo.Reals) #sum of congestion
    model_RD.total_costs = pyo.Var(within=pyo.NonNegativeReals)   #costs
    
        # Define Constraints 
    def def_congestion_1(m, l, t): #Congestion in case flow is positive
         capacity = df_lines.iloc[l]['capacity']
         return m.congestion[l, t] >= m.f[l, t] - capacity 
     
    model_RD.con_def_congestion_1 = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=def_congestion_1)
     
    def def_congestion_2(m, l, t):  #Congestion in case flow is negative
         capacity = df_lines.iloc[l]['capacity']
         return m.congestion[l, t] >= -m.f[l, t] - capacity 
     
    model_RD.con_def_congestion_2 = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=def_congestion_2)

     # DC powerflow equations
    def ref_theta(m, t):
         return m.theta[0, t] == 0.0
     
    model_RD.con_ref_theta = pyo.Constraint(model_RD.time_set, rule=ref_theta)
     
    def DC_flow(m, l, t):
         from_bus = df_lines.iloc[l]['from_bus'] 
         to_bus = df_lines.iloc[l]['to_bus'] 
         susceptances = df_lines.iloc[l]['susceptance']
         return m.f[l, t] == susceptances * (m.theta[from_bus, t] - m.theta[to_bus, t])
     
    model_RD.con_DC_flow = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=DC_flow)
     
    def nodal_power_balance(m, b, t):
        inflows = sum(m.f[l, t] for l in model_RD.line_set if df_lines.iloc[l]['to_bus'] == b)
        outflows = sum(m.f[l, t] for l in model_RD.line_set if df_lines.iloc[l]['from_bus'] == b)
        return inflows - outflows + m.p[b, t] == 0
    
    model_RD.con_nodal_power_balance = pyo.Constraint(model_RD.bus_set, model_RD.time_set, rule=nodal_power_balance)
    
    def def_p(m, b, t):
        # Find all assets corresponding to the bus b
        assets_at_bus = np.unique(df_RD_orderbook[df_RD_orderbook['bus'] == b]['asset'].values)
        
        # Sum over all assets corresponding to this bus
        dp_sum = sum(m.dp[asset, t] for asset in assets_at_bus)
        
        # Return the equation for p
        return m.p[b, t] == m.p_prognosis[b, t] + dp_sum

    model_RD.con_def_p = pyo.Constraint(model_RD.bus_set, model_RD.time_set, rule=def_p)
    
    def lower_bound_dp(m, a, t):
        return m.dp[a, t] >= m.max_dp_down[a, t]

    model_RD.con_lower_bound_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=lower_bound_dp)
    
        
    def upper_bound_dp(m, a, t):
        return m.dp[a,t ] <= m.max_dp_up[a, t]
    
    model_RD.con_upper_bound_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=upper_bound_dp)
    
    # Cleared buy = cleared sell
    def dp_balance(m, t):
        return sum(m.dp[a, t] for a in m.asset_set) == 0.0
    
    model_RD.con_dp_balance = pyo.Constraint( model_RD.time_set, rule=dp_balance)
    
    # Constraints to link `u` and `dp`. u is binary (0/1)  the following expresisons determine that if u is 1, dp has to be any value between 0 an 1e6
        #if u==0, dp is a negative price. This allow the objective funciton to add negative and positve prices. and allos for an upward bid and downward bid per location
    def link_u_dp_rule(m, a, t):
        return m.dp[a, t] <= m.u[a, t] * 1e5
    
    model_RD.link_u_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=link_u_dp_rule)
    
   
    def link_u_dp_neg_rule(m, a, t):
        return m.dp[a, t] >= -1e5 * (1 - m.u[a, t])
    
    model_RD.link_u_dp_neg = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=link_u_dp_neg_rule)
    
   
    def total_costs_def(m):
        return m.total_costs == sum(
            m.dp[a, t] * m.price_up[a, t] * m.u[a, t] +  
            m.dp[a, t] * m.price_down[a, t] * -(1 - m.u[a, t])  
            for a in m.asset_set for t in m.time_set
        )
    
    model_RD.total_costs_constraint = pyo.Constraint(rule=total_costs_def)
    
        # Add congestion defition
    def def_total_congestion(m):
        return m.total_congestion ==  sum(sum(m.congestion[l, t] * m.dt for l in m.line_set) for t in m.time_set)
    
    model_RD.con_def_total_congestion = pyo.Constraint(rule=def_total_congestion)
    
    '''
    #Objective function only costs
    def zero_congestion_constraint(m):
        return m.total_congestion == 0
    
    # Add the constraint to the model_RD
    model_RD.zero_congestion = pyo.Constraint(rule=zero_congestion_constraint)
    
    def objective_function(m):
        return m.total_costs
       
    
    '''
    #Objective function, congestion and costs mixxed
    
    epsilon = 0.0005
    def objective_function(m):
       return (1.0 - epsilon) * m.total_congestion + epsilon * m.total_costs
    
    
    #'''
    
    
    model_RD.objective_function = pyo.Objective(sense=pyo.minimize, expr=objective_function)
    
    
    # Store model_RD (convenient for debugging)
    path = os.getcwd() + '\\'
    model_RD.write(path + "model_RD_RD.lp", io_options={"symbolic_solver_labels": True})
    
    #%% Solve the model_RD
    opt = SolverFactory('gurobi')
    opt.options['mipgap'] = 0.001
    opt.options['DualReductions'] = 0
    results = opt.solve(model_RD, tee=Tee)
    
    termination_condition = results.solver.termination_condition
    '''
    if termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible.\n")
    elif termination_condition == pyo.TerminationCondition.unbounded:
        print("Model is unbounded.\n")
    elif termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
        print("Model is either infeasible or unbounded.\n")
    else:
        print(f"Termination condition: {termination_condition}\n")
        print(f"total costs for RD are {pyo.value(model_RD.total_costs)}\n")  
    '''
    #total_congestion_results = pyo.value(model_RD.total_congestion)
    #total_costs_results = pyo.value(model_RD.total_costs)    
    #print(total_costs_results)
    #print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
    #print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)} = {total_congestion_results * (np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)}\n")
      
    return model_RD, termination_condition

#%% Normal RD function (0 congestion)
def optimal_redispatch(load_per_node:pd.DataFrame, df_RD_orderbook:pd.DataFrame, Tee :int = 0) -> pyo.ConcreteModel:
    assets = list(np.unique(df_RD_orderbook.iloc[:,0].values))
    df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance)) #not a global parameter, so also have to do this function locally
    # Create a dictionary to map assets to their bus

    p_prognosis = load_per_node.copy()
    
    max_RD_up_dict = {}
    max_RD_down_dict = {}
    RD_price_up_dict = {}
    RD_price_down_dict = {}
    p_prognosis_dict = {index: value for index, value in np.ndenumerate(p_prognosis)}
    
    for asset in assets:  # Loop to construct the max_RD_up_dict
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'buy')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                max_power = filtered_df['power'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                max_RD_up_dict[key] = max_power
     
                        
    for asset in assets:  # Loop to construct the max_RD_down_dict
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'sell')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                max_power = -filtered_df['power'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                max_RD_down_dict[key] = max_power
    
    for asset in assets:  # Loop to construct the RD_price_up
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'buy')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                price_up = filtered_df['price'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                RD_price_up_dict[key] = price_up
                
                
    for asset in assets:  # Loop to construct the RD_price_down
        for t in range(0, ptus):
            key = (asset, t)
            if len(key) == 2:
                # Filter the DataFrame based on the asset, time range, and buy/sell condition
                filtered_df = df_RD_orderbook[
                    (df_RD_orderbook['asset'] == asset) & 
                    (df_RD_orderbook['delivery_start'] <= t) & 
                    (df_RD_orderbook['delivery_end'] > t) & 
                    (df_RD_orderbook['buy/sell'] == 'sell')
                ].fillna(0)  # Replace NaN with 0
    
                # Get the maximum 'power' for this asset and time period
                price_down = filtered_df['price'].max() if not filtered_df.empty else 0
    
                # Store the maximum power in the dictionary
                RD_price_down_dict[key] = price_down
         
    model_RD = pyo.ConcreteModel() # build the model_RD
   
    # Define sets (indices)
    model_RD.bus_set = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
    model_RD.line_set = pyo.RangeSet(0, n_lines - 1)
    model_RD.time_set = pyo.RangeSet(0, ptus - 1)  
    model_RD.asset_set = pyo.RangeSet(0, len(np.unique(df_RD_orderbook.iloc[:,0].values))-1)
    
    # Define parameters
    model_RD.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
    model_RD.max_dp_up = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=max_RD_up_dict)
    model_RD.price_up = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=RD_price_up_dict)
    model_RD.max_dp_down = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=max_RD_down_dict)
    model_RD.price_down = pyo.Param(model_RD.asset_set, model_RD.time_set, within=pyo.Reals, initialize=RD_price_down_dict)
    model_RD.p_prognosis = pyo.Param(model_RD.bus_set, model_RD.time_set, within=pyo.Reals, initialize=p_prognosis_dict)
    
    # Define variables
    model_RD.theta = pyo.Var(model_RD.bus_set, model_RD.time_set, within=pyo.Reals) # angle for DCLF
    model_RD.f = pyo.Var(model_RD.line_set, model_RD.time_set, within=pyo.Reals) #flow
    model_RD.congestion = pyo.Var(model_RD.line_set, model_RD.time_set, within=pyo.NonNegativeReals)  # >= 0
    model_RD.p = pyo.Var(model_RD.bus_set, model_RD.time_set, within=pyo.Reals)  # power
    model_RD.dp = pyo.Var(model_RD.asset_set, model_RD.time_set, within=pyo.Reals)  # dp after RD
    model_RD.u = pyo.Var(model_RD.asset_set, model_RD.time_set, within=pyo.Binary)  # Binary variable for condition
    
    
    model_RD.total_congestion = pyo.Var(within=pyo.Reals) #sum of congestion
    model_RD.total_costs = pyo.Var(within=pyo.Reals)   #costs
    
        # Define Constraints 
    def def_congestion_1(m, l, t): #Congestion in case flow is positive
         capacity = df_lines.iloc[l]['capacity']
         return m.congestion[l, t] >= m.f[l, t] - capacity 
     
    model_RD.con_def_congestion_1 = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=def_congestion_1)
     
    def def_congestion_2(m, l, t):  #Congestion in case flow is negative
         capacity = df_lines.iloc[l]['capacity']
         return m.congestion[l, t] >= -m.f[l, t] - capacity 
     
    model_RD.con_def_congestion_2 = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=def_congestion_2)
    '''
     # Define the zero congestion constraint
     def zero_congestion_constraint(m):
         return sum(m.congestion[l, t] for l in m.line_set for t in m.time_set) == 0
     
     # Add the constraint to the model_RD
     model_RD.zero_congestion = pyo.Constraint(rule=zero_congestion_constraint)
     '''
     # DC powerflow equations
    def ref_theta(m, t):
         return m.theta[0, t] == 0.0
     
    model_RD.con_ref_theta = pyo.Constraint(model_RD.time_set, rule=ref_theta)
     
    def DC_flow(m, l, t):
         from_bus = df_lines.iloc[l]['from_bus'] 
         to_bus = df_lines.iloc[l]['to_bus'] 
         susceptances = df_lines.iloc[l]['susceptance']
         return m.f[l, t] == susceptances * (m.theta[from_bus, t] - m.theta[to_bus, t])
     
    model_RD.con_DC_flow = pyo.Constraint(model_RD.line_set, model_RD.time_set, rule=DC_flow)
     
    def nodal_power_balance(m, b, t):
        inflows = sum(m.f[l, t] for l in model_RD.line_set if df_lines.iloc[l]['to_bus'] == b)
        outflows = sum(m.f[l, t] for l in model_RD.line_set if df_lines.iloc[l]['from_bus'] == b)
        return inflows - outflows + m.p[b, t] == 0
    
    model_RD.con_nodal_power_balance = pyo.Constraint(model_RD.bus_set, model_RD.time_set, rule=nodal_power_balance)
    
    def def_p(m, b, t):
        # Find all assets corresponding to the bus b
        assets_at_bus = np.unique(df_RD_orderbook[df_RD_orderbook['bus'] == b]['asset'].values)
        
        # Sum over all assets corresponding to this bus
        dp_sum = sum(m.dp[asset, t] for asset in assets_at_bus)
        
        # Return the equation for p
        return m.p[b, t] == m.p_prognosis[b, t] + dp_sum

    model_RD.con_def_p = pyo.Constraint(model_RD.bus_set, model_RD.time_set, rule=def_p)
    
    def lower_bound_dp(m, a, t):
        return m.dp[a, t] >= m.max_dp_down[a, t]

    model_RD.con_lower_bound_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=lower_bound_dp)
    
        
    def upper_bound_dp(m, a, t):
        return m.dp[a,t ] <= m.max_dp_up[a, t]
    
    model_RD.con_upper_bound_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=upper_bound_dp)
    
    # Cleared buy = cleared sell
    def dp_balance(m, t):
        return sum(m.dp[a, t] for a in m.asset_set) == 0.0
    
    model_RD.con_dp_balance = pyo.Constraint( model_RD.time_set, rule=dp_balance)
    
    # Constraints to link `u` and `dp`. u is binary (0/1)  the following expresisons determine that if u is 1, dp has to be any value between 0 an 1e6
        #if u==0, dp is a negative price. This allow the objective funciton to add negative and positve prices. and allos for an upward bid and downward bid per location
    def link_u_dp_rule(m, a, t):
        return m.dp[a, t] <= m.u[a, t] * 1e5
    
    model_RD.link_u_dp = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=link_u_dp_rule)
    
   
    def link_u_dp_neg_rule(m, a, t):
        return m.dp[a, t] >= -1e5 * (1 - m.u[a, t])
    
    model_RD.link_u_dp_neg = pyo.Constraint(model_RD.asset_set, model_RD.time_set, rule=link_u_dp_neg_rule)
    
   
    def total_costs_def(m):
        return m.total_costs == sum(
            m.dp[a, t] * m.price_up[a, t] * m.u[a, t] +  
            m.dp[a, t] * m.price_down[a, t] * -(1 - m.u[a, t])  
            for a in m.asset_set for t in m.time_set
        )
    
    model_RD.total_costs_constraint = pyo.Constraint(rule=total_costs_def)
    
        # Add congestion defition
    def def_total_congestion(m):
        return m.total_congestion ==  sum(sum(m.congestion[l, t] * m.dt for l in m.line_set) for t in m.time_set)
    
    model_RD.con_def_total_congestion = pyo.Constraint(rule=def_total_congestion)
    
    #'''
    #Objective function only costs
    def zero_congestion_constraint(m):
        return m.total_congestion == 0
    
    # Add the constraint to the model_RD
    model_RD.zero_congestion = pyo.Constraint(rule=zero_congestion_constraint)
    
    def objective_function(m):
        return m.total_costs
       
    
    '''
    #Objective function, congestion and costs mixxed
    
    epsilon = 0.0005
    def objective_function(m):
       return (1.0 - epsilon) * m.total_congestion + epsilon * m.total_costs
    
    
    '''
    
    
    model_RD.objective_function = pyo.Objective(sense=pyo.minimize, expr=objective_function)
    
    
    # Store model_RD (convenient for debugging)
    path = os.getcwd() + '\\'
    model_RD.write(path + "model_RD_RD.lp", io_options={"symbolic_solver_labels": True})
    
    #%% Solve the model_RD
    opt = SolverFactory('gurobi')
    opt.options['mipgap'] = 0.001
    opt.options['DualReductions'] = 0
    results = opt.solve(model_RD, tee=Tee)
    
    termination_condition = results.solver.termination_condition
    '''
    if termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible.\n")
    elif termination_condition == pyo.TerminationCondition.unbounded:
        print("Model is unbounded.\n")
    elif termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
        print("Model is either infeasible or unbounded.\n")
    else:
        print(f"Termination condition: {termination_condition}\n")
        print(f"total costs for RD are {pyo.value(model_RD.total_costs)}\n")  
    '''
    #total_congestion_results = pyo.value(model_RD.total_congestion)
    #total_costs_results = pyo.value(model_RD.total_costs)    
    #print(total_costs_results)
    #print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
    #print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)} = {total_congestion_results * (np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)}\n")
      
    return model_RD, termination_condition

#%% CBC function
def optimal_CBC(load_per_node:pd.DataFrame, df_CBC_orderbook:pd.DataFrame, congestion:float, chp_prog:pd.DataFrame, df_lines:pd.DataFrame = df_lines, Tee:int = 0) -> pyo.ConcreteModel:
    
   assets = list(np.unique(df_CBC_orderbook.iloc[:,0].values))
   df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance)) #not a global parameter, so also have to do this function locally
   # Create a dictionary to map assets to their bus
   
   p_prognosis = load_per_node.copy()
   
   max_CBC_up_dict = {}
   max_CBC_down_dict = {}
   CBC_price_up_dict = {}
   CBC_price_down_dict = {}
   p_prognosis_dict = {index: value for index, value in np.ndenumerate(p_prognosis)}
   
   for asset in assets:  # Loop to construct the max_CBC_up_dict
       for t in range(0, ptus):
           key = (asset, t)
           if len(key) == 2:
               # Filter the DataFrame based on the asset, time range, and buy/sell condition
               filtered_df = df_CBC_orderbook[
                   (df_CBC_orderbook['asset'] == asset) & 
                   (df_CBC_orderbook['delivery_start'] <= t) & 
                   (df_CBC_orderbook['delivery_end'] > t) & 
                   (df_CBC_orderbook['buy/sell'] == 'buy')
               ].fillna(0)  # Replace NaN with 0
   
               # Get the maximum 'power' for this asset and time period
               max_power = filtered_df['power'].max() if not filtered_df.empty else 0
   
               # Store the maximum power in the dictionary
               max_CBC_up_dict[key] = max_power
    
                       
   for asset in assets:  # Loop to construct the max_CBC_down_dict
       for t in range(0, ptus):
           key = (asset, t)
           if len(key) == 2:
               # Filter the DataFrame based on the asset, time range, and buy/sell condition
               filtered_df = df_CBC_orderbook[
                   (df_CBC_orderbook['asset'] == asset) & 
                   (df_CBC_orderbook['delivery_start'] <= t) & 
                   (df_CBC_orderbook['delivery_end'] > t) & 
                   (df_CBC_orderbook['buy/sell'] == 'sell')
               ].fillna(0)  # Replace NaN with 0
   
               # Get the maximum 'power' for this asset and time period
               max_power = -filtered_df['power'].max() if not filtered_df.empty else 0
   
               # Store the maximum power in the dictionary
               max_CBC_down_dict[key] = max_power
   
   for asset in assets:  # Loop to construct the CBC_price_up
       for t in range(0, ptus):
           key = (asset, t)
           if len(key) == 2:
               # Filter the DataFrame based on the asset, time range, and buy/sell condition
               filtered_df = df_CBC_orderbook[
                   (df_CBC_orderbook['asset'] == asset) & 
                   (df_CBC_orderbook['delivery_start'] <= t) & 
                   (df_CBC_orderbook['delivery_end'] > t) & 
                   (df_CBC_orderbook['buy/sell'] == 'buy')
               ].fillna(0)  # Replace NaN with 0
   
               # Get the maximum 'power' for this asset and time period
               price_up = filtered_df['price'].max() if not filtered_df.empty else 0
   
               # Store the maximum power in the dictionary
               CBC_price_up_dict[key] = price_up
               
               
   for asset in assets:  # Loop to construct the CBC_price_down
       for t in range(0, ptus):
           key = (asset, t)
           if len(key) == 2:
               # Filter the DataFrame based on the asset, time range, and buy/sell condition
               filtered_df = df_CBC_orderbook[
                   (df_CBC_orderbook['asset'] == asset) & 
                   (df_CBC_orderbook['delivery_start'] <= t) & 
                   (df_CBC_orderbook['delivery_end'] > t) & 
                   (df_CBC_orderbook['buy/sell'] == 'sell')
               ].fillna(0)  # Replace NaN with 0
   
               # Get the maximum 'power' for this asset and time period
               price_down = filtered_df['price'].max() if not filtered_df.empty else 0
   
               # Store the maximum power in the dictionary
               CBC_price_down_dict[key] = price_down
        
   model = pyo.ConcreteModel() # build the model
  
   # Define sets (indices)
   model.bus_set = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
   model.line_set = pyo.RangeSet(0, n_lines - 1)
   model.time_set = pyo.RangeSet(0, ptus - 1)  
   model.asset_set = pyo.RangeSet(0, len(np.unique(df_CBC_orderbook.iloc[:,0].values))-1)
   
   # Define parameters
   model.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
   model.max_dp_up = pyo.Param(model.asset_set, model.time_set, within=pyo.Reals, initialize=max_CBC_up_dict)
   model.price_up = pyo.Param(model.asset_set, model.time_set, within=pyo.Reals, initialize=CBC_price_up_dict)
   model.max_dp_down = pyo.Param(model.asset_set, model.time_set, within=pyo.Reals, initialize=max_CBC_down_dict)
   model.price_down = pyo.Param(model.asset_set, model.time_set, within=pyo.Reals, initialize=CBC_price_down_dict)
   model.p_prognosis = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=p_prognosis_dict)
   
   # Define variables
   model.theta = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals) # angle for DCLF
   model.f = pyo.Var(model.line_set, model.time_set, within=pyo.Reals) #flow
   model.congestion = pyo.Var(model.line_set, model.time_set, within=pyo.NonNegativeReals)  # >= 0
   model.p = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # power
   model.dp = pyo.Var(model.asset_set, model.time_set, within=pyo.Reals)  # dp after CBC
   model.u = pyo.Var(model.asset_set, model.time_set, within=pyo.Binary)
   model.balancing_p = pyo.Var(model.time_set, within=pyo.Reals)  # Power uesd for distributed slackbus
   
   
   model.total_congestion = pyo.Var(within=pyo.Reals) #sum of congestion
   model.total_costs = pyo.Var(within=pyo.Reals)   #costs
   
    
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
    
   
    # Define balancing_p (distributed slack over chp buses)
   def def_balancing_p(m, t):
       total_imbalance = sum(m.dp[a, t] for a in m.asset_set)
       n_chp = len(np.unique(chp_prog.iloc[:, 0].values))
       return m.balancing_p[t] == total_imbalance / n_chp
   
   model.con_def_balancing_p = pyo.Constraint(model.time_set, rule=def_balancing_p)
   
   
   def def_p(m, b, t):
        chp_busses = set(chp_prog.iloc[:, 0].values) 
        assets_at_bus = np.unique(df_CBC_orderbook[df_CBC_orderbook['bus'] == b]['asset'].values)
        
        balancing_term = m.balancing_p[t] if b in chp_busses else 0  # Ensure valid expression
    
        return m.p[b, t] == m.p_prognosis[b, t] + sum(m.dp[a, t] for a in assets_at_bus) - balancing_term
    
   model.con_def_p = pyo.Constraint(model.bus_set, model.time_set, rule=def_p)


   def lower_bound_dp(m, a, t):
        return m.dp[a, t] >= m.max_dp_down[a, t]

   model.con_lower_bound_dp = pyo.Constraint(model.asset_set, model.time_set, rule=lower_bound_dp)

    
   def upper_bound_dp(m, a, t):
        return m.dp[a,t ] <= m.max_dp_up[a, t]
    
   model.con_upper_bound_dp = pyo.Constraint(model.asset_set, model.time_set, rule=upper_bound_dp)
  
    
   def link_u_dp_rule(m, a, t):
        return m.dp[a, t] <= m.u[a, t] * 1e5
    
   model.link_u_dp = pyo.Constraint(model.asset_set, model.time_set, rule=link_u_dp_rule)
    
   
   def link_u_dp_neg_rule(m, a, t):
        return m.dp[a, t] >= -1e5 * (1 - m.u[a, t])
    
   model.link_u_dp_neg = pyo.Constraint(model.asset_set, model.time_set, rule=link_u_dp_neg_rule)


   def total_costs_def(m):
        return m.total_costs == sum(
            m.dp[a, t] * m.price_up[a, t] * m.u[a, t] +  
            m.dp[a, t] * m.price_down[a, t] * -(1 - m.u[a, t])  
            for a in m.asset_set for t in m.time_set
        )
    
   model.total_costs_constraint = pyo.Constraint(rule=total_costs_def)
      
   
   # Add congestion defition
   def def_total_congestion(m):
       return m.total_congestion ==  sum(sum(m.congestion[l, t] * m.dt for l in m.line_set) for t in m.time_set)
   
   model.con_def_total_congestion = pyo.Constraint(rule=def_total_congestion)
   
   '''
   #Objective function only costs
   def decrease_congestion_constraint(m):
       return m.total_congestion == congestion * (1 - ratio)
   
   model.decrease_congestion = pyo.Constraint(rule=decrease_congestion_constraint)
   
   def objective_function(m):
       return m.total_costs
   
 
   
   
   '''
   #Objective function, congestion and costs mixxed
   epsilon = 0.000005
   def objective_function(m):
      return (1.0 - epsilon) * m.total_congestion + epsilon * m.total_costs
   #'''
   
   
   
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
   '''
   if termination_condition == pyo.TerminationCondition.infeasible:
       print("\n Model is infeasible.\n")
   elif termination_condition == pyo.TerminationCondition.unbounded:
       sys.exit("Model is unbounded.\n")
   elif termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
       sys.exit("Model is either infeasible or unbounded.\n")
   else:
       print(f"Termination condition: {termination_condition}\n")
   
       #total_congestion_results = pyo.value(model.total_congestion)
       total_costs_results = pyo.value(model.total_costs)    
   #print(total_costs_results)
   #print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
   #print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)} = {total_congestion_results * (np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)}\n")
   print(f"total costs for CBC are {total_costs_results}\n") 
   '''
   return model, termination_condition