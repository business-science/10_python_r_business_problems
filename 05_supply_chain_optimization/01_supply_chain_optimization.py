# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 5: SUPPLY CHAIN OPTIMIZATION

import pulp

# Create a problem variable:
prob = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)

# Define decision variables:
production_units = pulp.LpVariable("Production_units", lowBound=0, cat='Continuous')
transport_units = pulp.LpVariable("Transport_units", lowBound=0, cat='Continuous')

# Define costs:
production_cost_per_unit = 10
transport_cost_per_unit = 5
storage_cost_per_unit = 2

# Objective function:
prob += production_cost_per_unit * production_units + transport_cost_per_unit * transport_units + storage_cost_per_unit * transport_units, "Total_Cost"

# Constraints:
prob += production_units <= 1000, "Max_Production_Capacity"
prob += transport_units <= production_units, "Transport_Less_Than_Production"
prob += transport_units >= 800, "Min_Demand_Fulfillment"

# Solve the problem:
prob.solve()

# Print the results:
print("Status:", pulp.LpStatus[prob.status])
print("Optimal Production units:", production_units.varValue)
print("Optimal Transport units:", transport_units.varValue)
print("Total Cost:", pulp.value(prob.objective))
