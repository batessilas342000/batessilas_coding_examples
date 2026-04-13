# Silas Bates
# Summer 2025
# CS 5800
# Linear Programming 
# 7/26/25
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Identifying all of the criteria from the question text
    cost_cereal = 1
    cost_meat = 2
    sale_frisky = 7
    sale_husky = 6
    packing_frisky_cost = 1.4
    packing_husky_cost = 0.6
    frisky_cereal = 1
    frisky_meat = 1.5
    husky_cereal = 2
    husky_meat = 1
    # Calculating the profit for each brand for the optimization equation
    profit_frisky = sale_frisky - (frisky_cereal * cost_cereal) - (frisky_meat * cost_meat)\
        - packing_frisky_cost
    
    profit_husky = sale_husky - (husky_cereal * cost_cereal) - (husky_meat * cost_meat)\
        - packing_husky_cost
    
    # Putting the profit as coefficients in the optimization
    # equation. Using negative to maximize instead of minimize
    c = [-profit_frisky, -profit_husky]

    # Constraint values from the problem text
    max_cereal = 240000
    max_meat = 180000
    max_production_frisky = 110000

    # A_ub is the coefficients for the constraint equations.
    # i.e. 1 pound cereal for Frisky Pup and 2 pounds for Husky
    # Hound. b_ub is the value of the inequality of the amount
    # of each resource available
    A_ub = [[1, 2], [1.5, 1]]
    b_ub = [max_cereal, max_meat]

    # Bounds set as the max number of each brand able to be 
    # produced in a month
    bounds = [(0, max_production_frisky), (0, None)]

    # Getting the results of the program
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    # Printing the results of the program
    if result.success:
        max_profit = -result.fun  # Negate because we minimized the negative flow
        print(f"Maximum Profit: {max_profit}")
    else:
        print("Optimization failed.")

    # Setting the linespace and constraint equations written in
    # y = mx + b form instead of y + x = b form
    x = np.linspace(-10, 150000, 20)
    constraint1 = 120000 - .5 * x
    constraint2 = 180000 - 1.5 * x
    max_x = 110000

    # Ploting the constraint lines and filling
    # in the feasability area
    plt.figure(figsize=(8,6))
    plt.plot(x, constraint1, label ="Cereal Constraint", color="blue")
    plt.plot(x, constraint2, label='Meat Constraint', color='red')
    plt.axvline(max_x, color='green', label='Maximum Frisky Pup Production')
    x_coords = [0, 110000, 60000, 0]
    y_coords = [0, 15000, 90000, 120000]
    plt.fill(x_coords, y_coords)
    plt.plot(60000, 90000, marker="o", color='red', label='Optimal Point')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title('Dog Food Maximization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()