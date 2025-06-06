# CBC_RD_thesis
 The model files used for simulating different CBC/RD activationstrategies for the completion of my master thesis at TU/eindhoven

 Files explanation
Set-up:
	-config
File to declare the susceptance, input file name, amount of PTUs, and the predetermiend CBC/RD ratio (in case you excute main_single)

	-config_loader
File acompanying -config.

	-NL_model_input
Input file used for the final modeling, The folder alternative inputs provides other filses you can use to test other scenarios or validate the model. (do change input file name in -config if you change input file)

	
Excecutables:
	-main_single
This file is used to perform a single iteration of a predetermined CBC/RD ratio 

	-main_function
The function derived from -main_single. this function is called by -main_iteration, and main_monte_carlo to perform the iterations

	-main_iteration
File used to perform single iterations of different CBC/RD ratios. Since it is single iterations you should set the MAPE to 0 to obtain reliable results

	-main_monte_carlo
File used to perform MC simulations (multiple iterations per RD/CBC ratio). 

Supportin files:
	- network_plotting
Contains functions used in -main_single to lot the networks, is not useful for large models.

	-RD_CBC_functions
Conatins the function that perform the optimisations to select the RD and CBC bids, using Pyomo and Gurobi, If you do not have a gurobi liscence,change opt = SolverFactory('gurobi') to opt = SolverFactory('GLPK'), or any other open-source solver engine	 


Alternative_strat_folder
This folder containt the files used to model the RD priority strategy. The files are named and structured in the same way as the files described above. 

