# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 5: SUPPLY CHAIN OPTIMIZATION

# Load the lpSolve package
library(lpSolve)

# Define the costs
production_cost_per_unit <- 10
transport_cost_per_unit <- 5  # Transportation cost
storage_cost_per_unit <- 2    # Storage cost

# Total transport cost including storage
total_transport_cost_per_unit <- transport_cost_per_unit + storage_cost_per_unit

# Objective function coefficients
objective <- c(production_cost_per_unit, total_transport_cost_per_unit)

# Constraints matrix
# Production units, Transport units
constraints <- matrix(c(1, 0,   # Production <= 1000 units
                        1, -1,  # Production units >= Transport units
                        0, 1),  # Transport units >= 800 units
                      nrow=3, byrow=TRUE)

# Right-hand side of the constraints
rhs <- c(1000, 0, 800)

# Directions of the constraints
directions <- c("<=", ">=", ">=")

# Solve the linear programming problem
solution <- lp("min", objective, constraints, directions, rhs)

# Output the results
print("Solution is optimal")
print(paste("Optimal Production units:", solution$solution[1]))
print(paste("Optimal Transport units:", solution$solution[2]))
print(paste("Total Cost:", solution$objval))

