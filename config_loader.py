# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:58:40 2025

@author: 304723
"""

# config_loader.py

import pandas as pd
import config

def retrieve_config_variables():
    """
    Dynamically assigns configuration variables and derived variables
    as separate variables and returns them.
    """
    # Static configuration variables
    ptus = config.ptus
    input_file = config.input_file
    susceptance = config.susceptance

    # Load data from input file
    df_lines = pd.read_excel(input_file, 'lines', header=0, index_col=0)

    # Compute derived variables
    buses = sorted(set(df_lines['from_bus']).union(set(df_lines['to_bus'])))
    n_buses = len(buses)
    n_lines = len(df_lines)

    # Return all variables
    return ptus, input_file, susceptance, df_lines, buses, n_buses, n_lines
