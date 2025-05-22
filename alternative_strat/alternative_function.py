# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:20:57 2025

@author: 304723
"""

# %%
def alt_function(safety_margin, i, old_margin,  noise_mape:float = 0.0 ) -> float:  
    
    from time import monotonic
    
    
    start_time = monotonic()
    
    
    import pandas as pd
    import numpy as np
    import sys
    import pyomo.environ as pyo
    
    
    # ### Load variables and input data
    # The input data needs to be a seperate Excel with specified lay-out
    # 
    # note: generation is negative and load is positive
    
    # %% initliase data
    #Determine whether or not the deterministic part has to be performed, this is not the case if the ame ratio is tested multiple times, like in the monte_carlo

    
    from config_loader import retrieve_config_variables
    
    # Retrieve variables as a tuple, use this line in sperate files to get teh same gloabl variables
    ptus, input_file, susceptance, df_lines, buses, n_buses, n_lines, ratio = retrieve_config_variables()
    
    df_loads_D2 = pd.read_excel(input_file,'loads', header=0, index_col=0)  #load D-2 prognoses for loads
    df_RE_D2 = pd.read_excel(input_file,'re', header=0, index_col=0)        #Load D-2 prognoses for renewable generation
    df_chp_max = pd.read_excel(input_file,'chp_max', header=0, index_col=0) #Load the maximum power output of CHPs
    
    #%% functions
    
    def add_to_array(df_in: pd.DataFrame, np_out: np.array) -> np.array:
        '''
        This is a function to add the values of one DF to another array, based on the nodes provided in the DF
    
        Parameters
        ----------
        df_in : pd.DataFrame
            DESCRIPTION.
        np_out : np.array
            DESCRIPTION.
    
        Returns
        -------
        np_out : TYPE
            DESCRIPTION.
    
        '''
        for idx, row in df_in.iterrows():  # Iterate over DataFrame rows
            node = row["node"]  # Use the "node" column explicitly
            if node in buses:  # Check if the node is in the buses list
                bus_idx = buses.index(node)  # Get the index of the node in the buses list
                values = row[1:].to_numpy(dtype=float)  # Convert the remaining columns to a NumPy array
                np_out[bus_idx, :] += values  # Add values to the corresponding row in np_out
        return np_out
    
    
    def CHP_dispatch_calc(df_chp_max: pd.DataFrame, load_per_node: pd.DataFrame) -> pd.DataFrame:
        '''
        Function to calculate how much CHP capacity has to be dispatched, simulation of market coupling
    
        Parameters
        ----------
        df_chp_max : pd.DataFrame
            DESCRIPTION.
        load_per_node : pd.DataFrame
            DESCRIPTION.
    
        Returns
        -------
        chp_df : TYPE
            DESCRIPTION.
    
        '''
        # Initialize an array for CHP dispatch
        chp_dispatch = np.zeros((len(df_chp_max), ptus))
        
        # Find total imbalance per PTU (power CHPs need to deliver to balance the grid)
        imbalance_per_ptu = load_per_node.sum(axis=0)
        p_chp_required = -imbalance_per_ptu  # Negative values indicate required generation
    
        # Loop over each PTU
        for t in range(ptus):
            p = p_chp_required[t]  # Power required for this PTU
            chp = 0  # Start with the first CHP unit
    
            # Balance the power for this PTU
            while p < 0.0 and chp < len(df_chp_max):  # Ensure we don't exceed available CHPs
                max_dispatch = df_chp_max.iloc[chp, 1]  # Max dispatch capacity for this CHP
                bus = df_chp_max.iloc[chp, 0]  # Bus associated with this CHP
    
                # Adjust max_dispatch if there's an order for this bus & PTU
                if not df_dp_CBC_order_level.empty:
                    for _, orders in df_dp_CBC_order_level.iterrows():
                        order, hour = orders['Index']  # Unpack tuple index
    
                        # Find the corresponding order in the orderbook
                        condition = (
                            (df_CBC_orderbook['type'] == 'CHP') & 
                            (df_CBC_orderbook['delivery_start'] == t) & 
                            (df_CBC_orderbook['bus'] == bus)
                        )
                        filtered_orders = df_CBC_orderbook.loc[condition].copy()  # Explicitly copy to avoid warnings
                        
                        if not filtered_orders.empty and orders['Index'][0] in filtered_orders.index:
                            max_dispatch = chp_prog.loc[chp, t]+orders['dp_value']
                           
                # Dispatch power from this CHP
                if abs(p) <= abs(max_dispatch):  # If this CHP can fully balance the power
                    chp_dispatch[chp, t] = p
                    p = 0  # Fully balanced
                else:  # Dispatch as much as possible from this CHP
                    chp_dispatch[chp, t] = max_dispatch
                    p -= max_dispatch  # Remaining imbalance
                    chp += 1  # Move to the next CHP
    
            p_chp_required[t] = p  # Update remaining imbalance for this PTU (should not become positive)
    
        # Convert to DataFrame and include node (bus) information
        chp_df = pd.DataFrame(chp_dispatch, columns=range(ptus))
        chp_df.insert(0, 'node', df_chp_max.iloc[:, 0])
    
        return chp_df
    
     
    def calculate_powerflow(df_loads_D2: np.array) -> pd.DataFrame:
        '''
        A function to calculate the powerflows according to DC powerflow approach
    
        Parameters
        ----------
        df_loads_D2 : np.array
            DESCRIPTION.
    
        Returns
        -------
        df_results : TYPE
            DESCRIPTION.
    
        '''
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
    
    
    def overload_calculation(df_flows : pd.DataFrame) -> pd.DataFrame:
        '''
        een functie die de overload kan bereken aan de hand van line capacity en flows
    
        Parameters
        ----------
        df_flows : pd.DataFrame
            DESCRIPTION.
    
        Returns
        -------
        df_congestion_D2 : TYPE
            DESCRIPTION.
    
        '''
        for line in df_congestion_D2.index:  # Iterate over line indices
            for t in range(ptus):  # Iterate over PTUs (time steps)
                overload = abs(df_flows.iloc[line,t]) - df_lines['capacity'].loc[line]
                if overload > 0:
                    df_congestion_D2.iloc[line, t] = float(overload) if df_flows.iloc[line,t]>0 else float(-overload)
                else:
                    df_congestion_D2.iloc[line, t] = 0
        return df_congestion_D2
    
    def merge_orders(orderbook:pd.DataFrame):
        """
        Merges consecutive orders for the same asset at the same bus if:
        - They have the same buy/sell type
        - They have the same price
        - They have the same power (volume)
        - They are consecutive in time
        
        Parameters:
        orderbook (pd.DataFrame): The original orderbook DataFrame.
    
        Returns:
        pd.DataFrame: The optimized orderbook with merged orders.
        """
        # Sort by key parameters to ensure proper merging
        orderbook = orderbook.sort_values(by=['asset', 'bus', 'buy/sell', 'price', 'power', 'delivery_start'])
    
        # List to store merged orders
        merged_orders = []
        
        # Start with the first order
        current_order = orderbook.iloc[0].copy()
    
        for i in range(1, len(orderbook)):
            row = orderbook.iloc[i]
    
            # Check if this order can be merged with the previous one
            if (
                row['asset'] == current_order['asset'] and
                row['bus'] == current_order['bus'] and
                row['buy/sell'] == current_order['buy/sell'] and
                row['price'] == current_order['price'] and
                row['power'] == current_order['power'] and
                row['delivery_start'] == current_order['delivery_end']  # Consecutive time
            ):
                # Extend the delivery period
                current_order['delivery_end'] = row['delivery_end']
            else:
                # Store the previous order and start a new one
                merged_orders.append(current_order)
                current_order = row.copy()
    
        # Add the last processed order
        merged_orders.append(current_order)
    
        return pd.DataFrame(merged_orders).reset_index(drop=True)
    
    def add_generation_to_orderbook(generation:pd.DataFrame,orderbook:pd.DataFrame,source:str)->pd.DataFrame:
        '''
        Funcion to add generation DFs to the orderbooks
    
        Parameters
        ----------
        generation : pd.DataFrame
            DESCRIPTION.
        orderbook : pd.DataFrame
            DESCRIPTION.
        source : str
            What label do you want to give to the genretation type (i.e. RE, CHP)
    
        Returns
        -------
        orderbook : TYPE
            DESCRIPTION.
    
        '''
        if len(orderbook)>0:
            asset = max(orderbook.iloc[:,0])+1
        else:
            asset = 0
        for index, row in generation.iloc[:, 1:].iterrows(): #This loop adds a bid for all expected RE production
            for t, value in row.items():
                if value < 0:
                    list_order = [[asset+index,generation.iloc[index, 0], t, t + 1,'buy',(prognosis_wholesale_price + CBC_RE_premium) if source == 'RE' 
                                   else (prognosis_wholesale_price + CBC_CHP_premium) if source == 'CHP' else prognosis_wholesale_price , -value, source]]
                    
                    orderbook = pd.concat([orderbook, pd.DataFrame(list_order, columns=orderbook.columns)], ignore_index=True)
        return orderbook
    
    def add_normal_noise(df_D2: pd.DataFrame, MAPE: float) -> pd.DataFrame:
        '''
        Add noise to a load or generation DF
    
        Parameters
        ----------
        df_D2 : pd.DataFrame
            DESCRIPTION.
        MAPE : float
            DESCRIPTION.
    
        Returns
        -------
        df_output : TYPE
            DESCRIPTION.
    
        '''
        df_output = df_D2.copy()
        
        for load in range(len(df_D2)):
            # Generate noise only for numerical columns (excluding 'node' column if present)
            num_cols = df_D2.columns[df_D2.columns != 'node']
            
            std = abs(1.25 * MAPE * np.mean(df_output.iloc[load, 1:]))       
            noise = np.random.normal(0, std, size=len(num_cols))
            noise = np.trunc(noise * 10**2) / 10**2  # Truncate to 2 decimal places
            
            # Get row values
            row_values = df_output.iloc[load, 1:]
            max_val = row_values.max()
            min_val = row_values.min()
            
            # Apply noise to all values except min, max, and zero values
            mask = (row_values != 0) & (row_values != max_val) & (row_values != min_val)
            modified_values = row_values + mask * noise  # Apply noise only where mask is True
            
            # Ensure values stay within the original range
            df_output.iloc[load, 1:] = np.clip(modified_values, min_val, max_val)
        
        return df_output
    
    # %% DC load-flow
    # First, we will use the the D-2 prognoses to perform a 'manual' DC load-flow. This load-flow will be used to visualise the network behaviour nd see where the congestion occurs. 
    
    #
    
    
    #Make a matrix where every node and every ptu have total load excluding CHP generation
    load_per_node_D2 = np.zeros((n_buses,ptus))
    
    
    # use the function to add RE and load profiels to the load per node
    load_per_node_D2 = add_to_array(df_RE_D2, load_per_node_D2)
    
    load_per_node_D2 = add_to_array(df_loads_D2, load_per_node_D2)
    
    #This is a function that identifies the imbalnce at every PTU and dispatches the CHPs to balance the system
        # The CHPs are dispatched in order, to mimic a merit order (highset = cheapest)
    global df_dp_CBC_order_level
    df_dp_CBC_order_level = pd.DataFrame()  
    
    
    
    #use both functions to add the CHP to the load per node.
    chp_prog = CHP_dispatch_calc(df_chp_max, load_per_node_D2)
    load_per_node_D2 = add_to_array(chp_prog, load_per_node_D2)
    
    
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
        
        
    
    
    #use function to create a DF with all the flows per line 
    
    df_flows_D2 = pd.DataFrame(columns = range(ptus))
    for t in range(ptus):
        df_flows_D2[t] = calculate_powerflow(load_per_node_D2[:,t])
    
    
    # ### Localise and deterimine congestion volume
    # Use the flow DF and the capacity of the lines to find the congestion
    
    # %% initieer een DF voor alle overloads
    df_congestion_D2 = pd.DataFrame(index=df_lines.index, columns = range(ptus))
    
    df_congestion_D2 = overload_calculation(df_flows_D2)
    congestion_D2 = sum(abs(df_congestion_D2).sum())
    
    #check if there is congstion in the system
    if congestion_D2 == 0:
        sys.exit('No congestion is the system\n')
        
    #%% Make prelim RD orderbook based on biedplichtcontracten
    #read orderbook
    df_RD_bp_orderbook = pd.read_excel(input_file,'RD', header=0) #read the orderbook
    df_RD_bp_orderbook = df_RD_bp_orderbook.iloc[0:0]
    
    #define prememiums and the expected D-1 price for price prognoses in the orederbooks
    prognosis_wholesale_price = df_chp_max.loc[len(chp_prog[chp_prog.iloc[:, 1:].sum(axis=1) < 0])-1,'price']
    CBC_RE_premium = 0.0*prognosis_wholesale_price #compensation on top off the wholesale market prognosis only missed profits from certificates
    CBC_CHP_premium = 0.2*prognosis_wholesale_price #compensation on top off the wholesale market prognosis for CHPs
    
    #Add the expected renewable generation
    df_RD_bp_orderbook = add_generation_to_orderbook(df_RE_D2, df_RD_bp_orderbook, 'RE') #renewable generation added to orderbook
    df_RD_bp_orderbook = add_generation_to_orderbook(chp_prog, df_RD_bp_orderbook, 'CHP') #CHP generation added to orderbook
    
    #change the pricing of RE and CHP RD orders
    for index, row in df_RD_bp_orderbook.iterrows():
        if row['type']=='RE':
            row['price'] = 5.0 #price of green certficates
            df_RD_bp_orderbook.iloc[index,:] = row
        if row['type']=='CHP' and row['buy/sell']=='buy':
            row['price'] = 20 #price of green fuel
            df_RD_bp_orderbook.iloc[index,:] = row
                                            
    #now we add the remainder of the CHP capacity to the orderbook for upward orders
    df_chp_remainder = chp_prog.copy()
    
    for row, values in df_chp_remainder.iterrows():
        df_chp_remainder.iloc[row, 1:] = values.iloc[1:].values - df_chp_max.loc[row, 'max']
    
    asset = max(df_RD_bp_orderbook.iloc[:,0])
    for row, values in df_chp_remainder.iterrows(): # aloop that adds all the remaining CHP capacity to the RD market
        asset += 1
        start = -1
        for p in values[1:].values:
            start += 1
            
            if p > 0:
                power = p
                price = df_chp_max.loc[row,'price']*1.1 #how much does an upward actio cost?
                bus = values['node']
    
                order_tuple = (asset, bus, start, start+1, 'sell', price, power, 'CHP')
                df_RD_bp_orderbook.loc[len(df_RD_bp_orderbook)] = order_tuple   
    
    df_RD_bp_orderbook = merge_orders(df_RD_bp_orderbook)
    
    #%% Use optimilisation to find what congestion is not/difficult to be solved by the epexcted RD posibilities
    if i == 0:
        from RD_CBC_functionsV3 import optimal_redispatch_congestion_cost
        global model_RD_prog
        global termination_condition_RD_prog
        model_RD_prog, termination_condition_RD_prog = optimal_redispatch_congestion_cost(load_per_node_D2,df_RD_bp_orderbook,0)
        
    #%% Get relevant data from model (dp and remaining congestion)
    total_costs_RD_prog = pyo.value(model_RD_prog.total_costs)
    
    #get delta power for eacht asset/bus
    dp_RD_prog_values = {
        (a, t): pyo.value(model_RD_prog.dp[a, t]) 
        for a in model_RD_prog.asset_set 
        for t in model_RD_prog.time_set
    }
    
    # Step 1: Extract asset-to-bus mapping from df_CBC_orderbook
    asset_to_bus = df_RD_bp_orderbook[['asset', 'bus']].drop_duplicates().set_index('asset')['bus']
    
    # Step 2: Aggregate dp values per bus
    dp_per_bus = {}
    
    for (a, t), dp_value in dp_RD_prog_values.items():
        bus = asset_to_bus[a]  # Get the bus corresponding to the asset
        if (bus, t) not in dp_per_bus:
            dp_per_bus[(bus, t)] = 0  # Initialize if not present
        dp_per_bus[(bus, t)] += dp_value  # Sum dp values per bus per time
    
    # Step 3: Convert aggregated dictionary to DataFrame
    df_RD_prog_dp = pd.DataFrame.from_dict(dp_per_bus, orient='index', columns=['dp'])
    df_RD_prog_dp.index = pd.MultiIndex.from_tuples(df_RD_prog_dp.index, names=['bus', 'time'])
    
    # Step 4: Pivot to have one row per bus, one column per time step
    df_RD_prog_dp = df_RD_prog_dp.unstack(level='time').fillna(0)
    df_RD_prog_dp.columns = df_RD_prog_dp.columns.droplevel()  # Remove extra level from columns
    
    # Step 5: Ensure all nodes (buses) are included, using `buses` list
    df_RD_prog_dp = df_RD_prog_dp.reindex(buses, fill_value=0)  # Include all buses
    
    # Step 6: Reset index and rename columns
    df_RD_prog_dp.reset_index(inplace=True)
    df_RD_prog_dp.rename(columns={'bus': 'node'}, inplace=True)  # Rename column
    
    #Get remanining congestion for each line
    congestion_RD_prog_values = {
         (l, t): pyo.value(model_RD_prog.congestion[l, t]) 
         for l in model_RD_prog.line_set 
         for t in model_RD_prog.time_set
     }
    
    #ONLY FOR TESTS ADD ARTIFICIAL PROBLEM VOLUME REMOVE LATER
    #congestion_RD_prog_values[15,0]=0.01
    print(f'Amount of congestion not sovlable by RD {sum(congestion_RD_prog_values.values())}')
    
    CBC_problem_volume = sum(congestion_RD_prog_values.values())
    
    
    #%% create a new fictionary line-capacity df that is based on the remanider of the congestion volume not solvable by RD
    
    safety_margin = safety_margin #How much does the flow in congested lines not salvable by RD have to be reduced on top of the congestion to account for pronogses error
    df_lines_CBC = df_lines.copy()
    
    df_lines_CBC.loc[:,'capacity'] = 9999
    for (l,t) in congestion_RD_prog_values:
        if congestion_RD_prog_values[l,t] != 0 :
            df_lines_CBC.loc[l,'capacity'] = abs(df_flows_D2.iloc[l,t])- (congestion_RD_prog_values[l,t]*(1+safety_margin))
            
            
    # ### Activate CBCs
    # Het Idee is dat eerst CBCs worden afgeroepen (optimalisereend voor de kosten en het congestievolume terugdringend naar een bepaald niveau).
    # Vervolgens word er weer gedispatched met als constraint dat he tcongestievolume niet mag toenemen
    
    # %%
    df_CBC_orderbook = pd.read_excel(input_file,'CBC', header=0) #read the orderbook
    
    #add all activated RE and CHP to the CBC orderbook, in order to do this, an assumption about the D-1 price is made
    prognosis_wholesale_price = df_chp_max.loc[len(chp_prog[chp_prog.iloc[:, 1:].sum(axis=1) < 0])-1,'price']
    CBC_RE_premium = 0.0*prognosis_wholesale_price #compensation on top off the wholesale market prognosis only missed profits from certificates
    CBC_CHP_premium = 0.2*prognosis_wholesale_price #compensation on top off the wholesale market prognosis for CHPs
    
    
    df_CBC_orderbook = add_generation_to_orderbook(df_RE_D2, df_CBC_orderbook, 'RE')
    
    df_CBC_orderbook = merge_orders(df_CBC_orderbook)

    start_time = monotonic()
    if i == 0 or safety_margin != old_margin:
        from RD_CBC_functionsV3 import optimal_CBC
        global model_CBC
        global termination_condition_CBC
        model_CBC, termination_condition_CBC = optimal_CBC(load_per_node_D2, df_CBC_orderbook,congestion_D2, chp_prog, df_lines_CBC)
        #print(f"Run time of CBC function {monotonic() - start_time} seconds\n")
        if termination_condition_CBC == pyo.TerminationCondition.infeasible:
            return [0,0,0,0,0,safety_margin]
    
    total_costs_CBC = pyo.value(model_CBC.total_costs)
    
    
    dp_CBC_values = {
        (a, t): pyo.value(model_CBC.dp[a, t]) 
        for a in model_CBC.asset_set 
        for t in model_CBC.time_set
    }
    
    # Step 1: Extract asset-to-bus mapping from df_CBC_orderbook
    asset_to_bus = df_CBC_orderbook[['asset', 'bus']].drop_duplicates().set_index('asset')['bus']
    
    # Step 2: Aggregate dp values per bus
    dp_per_bus = {}
    
    for (a, t), dp_value in dp_CBC_values.items():
        bus = asset_to_bus[a]  # Get the bus corresponding to the asset
        if (bus, t) not in dp_per_bus:
            dp_per_bus[(bus, t)] = 0  # Initialize if not present
        dp_per_bus[(bus, t)] += dp_value  # Sum dp values per bus per time
    
    # Step 3: Convert aggregated dictionary to DataFrame
    df_dp = pd.DataFrame.from_dict(dp_per_bus, orient='index', columns=['dp'])
    df_dp.index = pd.MultiIndex.from_tuples(df_dp.index, names=['bus', 'time'])
    
    # Step 4: Pivot to have one row per bus, one column per time step
    df_dp_CBC = df_dp.unstack(level='time').fillna(0)
    df_dp_CBC.columns = df_dp_CBC.columns.droplevel()  # Remove extra level from columns
    
    # Step 5: Ensure all nodes (buses) are included, using `buses` list
    df_dp_CBC = df_dp_CBC.reindex(buses, fill_value=0)  # Include all buses
    
    # Step 6: Reset index and rename columns
    df_dp_CBC.reset_index(inplace=True)
    df_dp_CBC.rename(columns={'bus': 'node'}, inplace=True)  # Rename column
    
    
    # ### Actualise prognoses
    # Introduce 'noise' to the prognoses so they represent the actual T-pofile data to be used for the marketcoupling later the CBC will also be taken into consoderation during this stage, but not yet implemented
    
    # %%
    # add noise to the D-2 to make 'actual' data. use these to make new load per node, and new CHP dispatch (marketcoupling). Results in a new load per node and new congestion 
    df_loads = add_normal_noise(df_loads_D2,noise_mape*(30/21))
    df_RE    = add_normal_noise(df_RE_D2, noise_mape)
    
    load_per_node = np.zeros((n_buses,ptus))
    load_per_node = add_to_array(df_RE, load_per_node)
    load_per_node = add_to_array(df_loads, load_per_node)
    load_per_node = add_to_array(df_dp_CBC, load_per_node)
    
    chp_coupling = CHP_dispatch_calc(df_chp_max, load_per_node)
    #find total load per node per CHP 
    
    load_per_node = add_to_array(chp_coupling, load_per_node)
    
    
    #calculate congestion after noise was introduced
    df_flows = pd.DataFrame(columns = range(ptus))
    for t in range(ptus):
        df_flows[t] = calculate_powerflow(load_per_node[:,t])
        
    df_congestion = overload_calculation(df_flows)
    congestion = sum(abs(df_congestion).sum())
    
    
    #Calculate how much is paid for electricity in the entire market
    costs_market = 0
    for t, column in chp_coupling.iloc[:,1:].iteritems():
        #print(t)
        merit = len(column[column!=0])
        market_price = df_chp_max.iloc[merit-1,2]
        cost = market_price * df_loads.iloc[:,t+1].sum()
        costs_market += cost
    market_price = costs_market/sum(df_loads.iloc[:,1:].sum())
        

    # %% Vsualization of generation per type and per node before RD
    
    #make a dict with all the load per generation type
    load_per_type = {'df_loads': ptus * [0], 'df_RE': ptus * [0], 'chp_coupling': ptus * [0], 'df_dp_CBC': ptus * [0]}
    for key in load_per_type.keys():
        load_per_type[key] = locals()[key].iloc[:,1:].sum()
        
    load_per_type = pd.DataFrame(load_per_type)
    load_per_type['imbalance'] = load_per_type.sum(axis=1)
    
    
    # ### Redispatch
    # Using Pyomo, and load-flow constraints, dispatch the optimal set of bids to minimze costs and mitigate any remaining congestion. 
    # Input shouldbe a balanced DF wih load per node per ptu
    
    # %%
    #sys.exit('stop before RD\n')
    
    #read orderbook
    df_RD_orderbook = pd.read_excel(input_file,'RD', header=0) #read the orderbook
    
    #Add the renewable generation that is not CBC'd
    df_dp_CBC_pos = df_dp_CBC.copy()  # Ensure we don't modify the original DataFrame
    df_dp_CBC_pos.iloc[:, 1:] = df_dp_CBC.iloc[:, 1:].where(df_dp_CBC.iloc[:, 1:] > 0, 0) #now we have the CBC DF of oly the CBCs for generation
    
    np_RE_CBC = np.zeros((n_buses,ptus))
    np_RE_CBC = add_to_array(df_RE,np_RE_CBC)
    np_RE_CBC = add_to_array(df_dp_CBC_pos,np_RE_CBC) #an array with the limited RE generation
    np_RE_CBC[np_RE_CBC > 0] = 0
    df_RE_CBC = df_dp_CBC_pos.copy()
    df_RE_CBC.iloc[:,1:] = np_RE_CBC
    
    df_RE_CBC = df_RE_CBC.loc[~(df_RE_CBC.iloc[:, 1:] == 0).all(axis=1)].reset_index(drop=True)
    df_RD_orderbook = add_generation_to_orderbook(df_RE_CBC, df_RD_orderbook, 'RE') #renewable generation added to orderbook
    df_RD_orderbook = add_generation_to_orderbook(chp_coupling, df_RD_orderbook, 'CHP') #CHP generation added to orderbook
    
    #change the pricing of RE and CHP RD orders
    for index, row in df_RD_orderbook.iterrows():
        if row['type']=='RE':
            row['price'] = 6.0 #price of green certficates
            df_RD_orderbook.iloc[index,:] = row
        if row['type']=='CHP' and row['buy/sell']=='buy':
            row['price'] = -20 #price of green certficates
            df_RD_orderbook.iloc[index,:] = row
                                            
    #now we add the remainder of the CHP capacity to the orderbook for upward orders
    df_chp_remainder = chp_coupling.copy()
    
    for row, values in df_chp_remainder.iterrows():
        df_chp_remainder.iloc[row, 1:] = values.iloc[1:].values - df_chp_max.loc[row, 'max']
    
    asset = max(df_RD_orderbook.iloc[:,0])
    for row, values in df_chp_remainder.iterrows(): # aloop that adds all the remaining CHP capacity to the RD market
        asset += 1
        start = -1
        for p in values[1:].values:
            start += 1
            
            if p > 0:
                power = p
                price = df_chp_max.loc[row,'price']*1.1 #how much does an upward actio cost?
                bus = values['node']
    
                order_tuple = (asset, bus, start, start+1, 'sell', price, power, 'CHP')
                df_RD_orderbook.loc[len(df_RD_orderbook)] = order_tuple   
    
    df_RD_orderbook = merge_orders(df_RD_orderbook)
    start_time = monotonic()
    # something
    from RD_CBC_functionsV3 import optimal_redispatch
    
    model_RD, termination_condition_RD = optimal_redispatch(load_per_node, df_RD_orderbook)
    if termination_condition_RD == pyo.TerminationCondition.infeasible:
        return [0,0,0,0,0,safety_margin]


    
    # %%
    
    dp_RD_asset_level = {
        (a, t): pyo.value(model_RD.dp[a, t]) 
        for a in model_RD.asset_set 
        for t in model_RD.time_set
    }
    
    # Step 1: Extract asset-to-bus mapping from df_RD_orderbook
    asset_to_bus = df_RD_orderbook[['asset', 'bus']].drop_duplicates().set_index('asset')['bus']
    
    # Step 2: Aggregate dp values per bus
    dp_per_bus = {}
    
    for (a, t), dp_value in dp_RD_asset_level.items():
        
        if a not in asset_to_bus:
            bus = 0
        else:    
            bus = asset_to_bus[a]  # Get the bus corresponding to the asset
        if (bus, t) not in dp_per_bus:
            dp_per_bus[(bus, t)] = 0  # Initialize if not present
        dp_per_bus[(bus, t)] += dp_value  # Sum dp values per bus per time
    
    # Step 3: Convert aggregated dictionary to DataFrame
    df_dp = pd.DataFrame.from_dict(dp_per_bus, orient='index', columns=['dp'])
    df_dp.index = pd.MultiIndex.from_tuples(df_dp.index, names=['bus', 'time'])
    
    # Step 4: Pivot to have one row per bus, one column per time step
    df_dp_RD = df_dp.unstack(level='time').fillna(0)
    df_dp_RD.columns = df_dp_RD.columns.droplevel()  # Remove extra level from columns
    
    # Step 5: Ensure all nodes (buses) are included, using `buses` list
    df_dp_RD = df_dp_RD.reindex(buses, fill_value=0)  # Include all buses
    
    # Step 6: Reset index and rename columns
    df_dp_RD.reset_index(inplace=True)
    df_dp_RD.rename(columns={'bus': 'node'}, inplace=True)  # Rename column
    
    
    
    total_costs_RD = pyo.value(model_RD.total_costs)
    
    
    total_costs = total_costs_RD + total_costs_CBC
    print(f'\nTotal costs are {total_costs}.\n-RD costs = {total_costs_RD} \n-CBC costs = {total_costs_CBC}')
    old_margin = safety_margin

    return total_costs_CBC, total_costs_RD, total_costs, costs_market, market_price, old_margin
    
    
    
    
    
