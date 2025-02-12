import gurobipy as gp
import os


def write_iis(path: str, file_name: str):
    gurobi_model = gp.read(path + file_name)
    gurobi_model.optimize()
    # Check if the model is infeasible
    print(f"Status: {gurobi_model.Status}")

    if gurobi_model.Status == 4:
        print("Model is Unbounded!")


    if gurobi_model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
        print("Model is infeasible. Computing IIS...")

        # Compute the IIS
        gurobi_model.computeIIS()

        # Write the IIS to a file
        gurobi_model.write(path + "ISS.ilp")

        # Read and print the IIS file
        with open(path + "ISS.ilp", "r") as iis_file:
            iis = iis_file.read()
        print("IIS:\n", iis)

    else:
        print("Model is not infeasible. No need to compute IIS")

if __name__ == "__main__":
    path = os.getcwd() + '\\'
    file_name = 'model_CBC.lp'
    write_iis(path, file_name)