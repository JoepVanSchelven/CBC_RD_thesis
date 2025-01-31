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
import sys
import os

    # Retrieve variables as a tuple, use this line in sperate files to get teh same gloabl variables
ptus, input_file, susceptance, df_lines, buses, n_buses, n_lines, ratio = retrieve_config_variables()


def optimal_redispatch(load_per_node:pd.DataFrame, df_RD_orderbook:pd.DataFrame, Tee :int = 0) -> pyo.ConcreteModel:
    
    df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance))
    max_redispatch_up       = np.zeros((n_buses, ptus)) #the maximum increase in power (more consumption/less production) at a given location
    max_redispatch_down     = np.zeros((n_buses, ptus)) #the maximum decrease in power (less consumption/more production) at a given location
    redispatch_price_up     = np.zeros((n_buses, ptus))
    redispatch_price_down   = np.zeros((n_buses, ptus))
    p_prognosis             = load_per_node.copy()
    

    
    
    for _, row in df_RD_orderbook.iterrows():
        new_power = row['power'] if row['buy/sell'] == 'buy' else - row['power']
        if new_power > 0:   # Create an array with the maximum upward per per time/octino and a dict with the corresponding price
            max_redispatch_up[row['bus'], row['delivery_start']: row['delivery_end']] += new_power
            redispatch_price_up[row['bus'], row['delivery_start']: row['delivery_end']] = row['price']
    
        elif new_power < 0:     # Create an array with the maximum downward per per time/octino and a dict with the corresponding price 
            max_redispatch_down[row['bus'], row['delivery_start']: row['delivery_end']] += new_power
            redispatch_price_down[row['bus'], row['delivery_start'] : row['delivery_end']] = row['price']
    
    # Convert to dicts for pyomo initialization
    max_redispatch_up_dict      = {index: value for index, value in np.ndenumerate(max_redispatch_up)}
    max_redispatch_down_dict    = {index: value for index, value in np.ndenumerate(max_redispatch_down)}
    redispatch_price_up_dict    = {index: value for index, value in np.ndenumerate(redispatch_price_up)}
    redispatch_price_down_dict  = {index: value for index, value in np.ndenumerate(redispatch_price_down)}
    p_prognosis_dict            = {index: value for index, value in np.ndenumerate(p_prognosis)}
     
    
    model = pyo.ConcreteModel() # build the model
    
    # Define sets (indices)
    model.bus_set   = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
    model.line_set  = pyo.RangeSet(0, n_lines - 1)
    model.time_set  = pyo.RangeSet(0, ptus - 1)  
    
    # Define parameters
    model.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
    model.max_dp_up     = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=max_redispatch_up_dict)
    model.price_up      = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=redispatch_price_up_dict)
    model.max_dp_down   = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=max_redispatch_down_dict)
    model.price_down    = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=redispatch_price_down_dict)
    model.p_prognosis   = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=p_prognosis_dict)
    
    # Define variables
    model.theta      = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals) # angle for DCLF
    model.f          = pyo.Var(model.line_set, model.time_set, within=pyo.Reals) #flow
    model.congestion = pyo.Var(model.line_set, model.time_set, within=pyo.NonNegativeReals)  # >= 0
    model.p          = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # power
    model.dp         = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # dp after redispatch
    model.u          = pyo.Var(model.bus_set, model.time_set, within=pyo.Binary)  # Binary variable for condition
    
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
    
    # Define p
    def def_p(m, b, t):
        return m.p[b, t] == m.p_prognosis[b, t] + m.dp[b, t]
    
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
    
    # Cleared buy = cleared sell
    def dp_balance(m, t):
        return sum(m.dp[b, t] for b in m.bus_set) == 0.0
    
    model.con_dp_balance = pyo.Constraint( model.time_set, rule=dp_balance)
    
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
    def zero_congestion_constraint(m):
        return m.total_congestion == 0
    
    # Add the constraint to the model
    model.zero_congestion = pyo.Constraint(rule=zero_congestion_constraint)
    
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
    model.write(path + "model_RD.lp", io_options={"symbolic_solver_labels": True})
    
    #%% Solve the model
    opt = SolverFactory('gurobi')
    opt.options['mipgap'] = 0.001
    opt.options['DualReductions'] = 0
    results = opt.solve(model, tee=Tee)
    
    termination_condition = results.solver.termination_condition
    
    if termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible.\n")
    elif termination_condition == pyo.TerminationCondition.unbounded:
        print("Model is unbounded.\n")
    elif termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
        print("Model is either infeasible or unbounded.\n")
    else:
        print(f"Termination condition: {termination_condition}\n")
    
    #total_congestion_results = pyo.value(model.total_congestion)
    #total_costs_results = pyo.value(model.total_costs)    
    #print(total_costs_results)
    #print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
    #print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)} = {total_congestion_results * (np.mean(redispatch_price_up[redispatch_price_up > 0]) if np.any(redispatch_price_up > 0) else 0 + np.mean(redispatch_price_down[redispatch_price_down > 0]) if np.any(redispatch_price_down > 0) else 0)}\n")
    print(f"total costs for RD are {pyo.value(model.total_costs)}\n")    
    return model

#%% CBC function
def optimal_CBC(load_per_node:pd.DataFrame, df_CBC_orderbook:pd.DataFrame, congestion:float, Tee:int = 0, ratio: int= ratio) -> pyo.ConcreteModel:
    
    df_lines['susceptance'] = 1/(df_lines['len']*(1/susceptance)) #not a global parameter, so also have to do this function locally

    p_prognosis = load_per_node.copy()

    # Convert to dicts for pyomo initialization
    max_CLC_up_dict = {}
    max_CLC_down_dict = {}
    CLC_price_up_dict = {}
    CLC_price_down_dict = {}
    p_prognosis_dict = {index: value for index, value in np.ndenumerate(p_prognosis)}
     
    for bus in buses: #loop to construct the max_CLC_up_dict
        for t in range(0,ptus):
            for order,row in df_CBC_orderbook.iterrows():
                key = (bus,t,order)
                if len(key)==3:
                    max_CLC_up_dict[key] = row['power'] if row['power']>=0 and row['bus'] == bus and row['buy/sell']=='buy' and row['delivery_start']<=t<row['delivery_end'] else 0

    for bus in buses: #loop to construct the max_CLC_down_dict
        for t in range(0,ptus):
            for order,row in df_CBC_orderbook.iterrows():
                key = (bus,t,order)
                if len(key)==3:
                    max_CLC_down_dict[key] = -row['power'] if row['power']>=0 and row['bus'] == bus and row['buy/sell']=='sell' and row['delivery_start']<=t<row['delivery_end'] else 0

    for bus in buses: #loop to construct the CLC_price_down_dict
        for t in range(0,ptus):
            for order,row in df_CBC_orderbook.iterrows():
                key = (bus,t,order)
                if len(key)==3:
                    CLC_price_down_dict[key] = row['price'] if row['price']<=0 and row['bus'] == bus and row['delivery_start']<=t<row['delivery_end'] else 0

    for bus in buses: #loop to construct the CLC_price_up_dict
        for t in range(0,ptus):
            for order,row in df_CBC_orderbook.iterrows():
                key = (bus,t,order)
                if len(key)==3:
                    CLC_price_up_dict[key] = row['price'] if row['price']>=0 and row['bus'] == bus and row['delivery_start']<=t<row['delivery_end'] else 0

    
    model = pyo.ConcreteModel() # build the model
    # Define sets (indices)
    model.bus_set = pyo.RangeSet(0, n_buses - 1) #includes final value unlike normal python (hence -1)
    model.line_set = pyo.RangeSet(0, n_lines - 1)
    model.time_set = pyo.RangeSet(0, ptus - 1)  
    model.order_set = pyo.RangeSet(0,len(df_CBC_orderbook)-1)
    
    # Define parameters
    model.dt = pyo.Param(initialize=1.0)  # 1 hour time resolution
    model.max_dp_up = pyo.Param(model.bus_set, model.time_set, model.order_set, within=pyo.Reals, initialize=max_CLC_up_dict)
    model.price_up = pyo.Param(model.bus_set, model.time_set, model.order_set, within=pyo.Reals, initialize=CLC_price_up_dict)
    model.max_dp_down = pyo.Param(model.bus_set, model.time_set,model.order_set, within=pyo.Reals, initialize=max_CLC_down_dict)
    model.price_down = pyo.Param(model.bus_set, model.time_set, model.order_set, within=pyo.Reals, initialize=CLC_price_down_dict)
    model.p_prognosis = pyo.Param(model.bus_set, model.time_set, within=pyo.Reals, initialize=p_prognosis_dict)
    
    # Define variables
    model.theta = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals) # angle for DCLF
    model.f = pyo.Var(model.line_set, model.time_set, within=pyo.Reals) #flow
    model.congestion = pyo.Var(model.line_set, model.time_set, within=pyo.NonNegativeReals)  # >= 0
    model.p = pyo.Var(model.bus_set, model.time_set, within=pyo.Reals)  # power
    model.dp = pyo.Var(model.bus_set, model.time_set, model.order_set, within=pyo.Reals)  # dp after CLC
    model.balancing_p = pyo.Var(model.time_set, within=pyo.Reals)  # Power uesd for distributed slackbus
    model.u = pyo.Var(model.bus_set, model.time_set, within=pyo.Binary)  # Binary variable for condition
    
    model.total_congestion = pyo.Var(within=pyo.Reals) #sum of congestion
    model.total_costs = pyo.Var(within=pyo.Reals)   #costs
    
    
    # Constraints to link `u` and `dp`. u is binary (0/1)  the following expresisons determine that if u is 1, dp has to be any value between 0 an 1e6
        #if u==0, dp is a negative price. This allow the objective funciton to add negative and positve prices. and allos for an upward bid and downward bid per location
    def link_u_dp_rule(m, b, t, o):
        return m.dp[b, t, o] <= m.u[b, t] * 1e9  # Arbitrarily large value             dp is kliener dan 0 (u=0) of 1e9 (u=1)
    model.link_u_dp = pyo.Constraint(model.bus_set, model.time_set, model.order_set, rule=link_u_dp_rule)
    
    def link_u_dp_neg_rule(m, b, t, o):
        return m.dp[b, t, o] >= -1e9 * (1 - m.u[b, t])  # Arbitrarily large value      dp is groter dan -1e9 (u=0) of 0 (u=1)
    model.link_u_dp_neg = pyo.Constraint(model.bus_set, model.time_set, model.order_set, rule=link_u_dp_neg_rule)
    
    
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
    
    # Define balancing_p (distributed slack)
    def def_balancing_p(m, t):
        total_imbalance = sum(m.dp[b, t, o] for b in m.bus_set for o in m.order_set)
        n_buses = len(m.bus_set)
        return m.balancing_p[t] == total_imbalance / n_buses
    
    model.con_def_balancing_p = pyo.Constraint(model.time_set, rule=def_balancing_p)
    
    def def_p(m, b, t):
        return m.p[b, t] == m.p_prognosis[b, t] + sum(m.dp[b, t, o] for o in m.order_set) - m.balancing_p[t]
    
    model.con_def_p = pyo.Constraint(model.bus_set, model.time_set, rule=def_p)
    
    # Bound dp from below and above
    def lower_bound_dp(m, b, t, o):
        if m.max_dp_down[b, t, o] >= 0:
            return m.dp[b, t, o] >= 0
        else:
            return m.dp[b, t, o] >= m.max_dp_down[b, t, o]
        
    model.con_lower_bound_dp = pyo.Constraint(model.bus_set, model.time_set, model.order_set, rule=lower_bound_dp)
        
    def upper_bound_dp(m, b, t, o):
        if m.max_dp_up[b, t, o] >= 0:
            return m.dp[b, t, o] <= m.max_dp_up[b, t, o]
        else:
            return m.dp[b, t, o] <= 0.0
    
    model.con_upper_bound_dp = pyo.Constraint(model.bus_set, model.time_set, model.order_set, rule=upper_bound_dp)
    
    
    # Objective function
    def total_costs_def(m):
        return m.total_costs == sum(
            m.dp[b, t, o] * (m.price_up[b, t, o] * m.u[b, t] + m.price_down[b, t, o] * (1 - m.u[b, t])) * m.dt
            for b in m.bus_set
            for t in m.time_set
            for o in m.order_set
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
    #print(total_costs_results)
    #print(f"Cost term is {round(100.0 * total_costs_results * epsilon/of_results, 2)}% of the total OF (should be small)\n")
    #print(f"congestion volume X avg_Price = {total_congestion_results} X {(np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)} = {total_congestion_results * (np.mean(CLC_price_up[CLC_price_up > 0]) if np.any(CLC_price_up > 0) else 0 + np.mean(CLC_price_down[CLC_price_down > 0]) if np.any(CLC_price_down > 0) else 0)}\n")
    print(f"total costs for CBC are {total_costs_results}\n") 
    return model