CBC_RD_Thesis

This repository contains the model files used to simulate different CBC/RD activation strategies for the completion of my Master’s thesis at TU/eindhoven.
📁 File Overview
Set-up Files

    -config
    Defines key model parameters such as susceptance, the input file name, the number of PTUs, and the predetermined CBC/RD ratio (used when executing main_single).

    -config_loader
    Supporting script that loads the parameters defined in -config.

    NL_model_input
    The primary input file used for the final modeling.
    Note: The alternative_inputs/ folder contains additional input files for testing other scenarios. If you switch input files, make sure to update the file name in -config.

Executable Files

    main_single
    Runs a single iteration based on a fixed CBC/RD ratio.

    main_function
    Core function extracted from main_single. It is called by both main_iteration and main_monte_carlo to execute simulations.

    main_iteration
    Executes a series of single-iteration simulations using varying CBC/RD ratios.
    Important: To ensure reliable results, set the MAPE to 0.

    main_monte_carlo
    Runs Monte Carlo simulations — multiple iterations for each CBC/RD ratio.

Supporting Files

    network_plotting
    Contains functions for visualizing network configurations, used by main_single.
    Note: Not suitable for large network models.

    RD_CBC_functions
    Implements the optimization logic for selecting RD and CBC bids using Pyomo and Gurobi.
    Tip: If you don’t have a Gurobi license, replace SolverFactory('gurobi') with SolverFactory('glpk') or any other supported open-source solver.

📁 Alternative_strat_folder/

Contains files used to model the RD-priority strategy.
These files follow the same naming and structure conventions as those described above.
