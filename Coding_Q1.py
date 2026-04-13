# Silas Bates
# Summer 2025
# CS 5800
# Linear Programming 
# 7/26/25

from scipy.optimize import linprog
import numpy as np

def MaxFlow (capacity_matrix, source, sink):

    num_nodes = capacity_matrix.shape[0]

    edges = []
    capacities = []

    # Filling out the edges and capacities arrays
    # to be used later in the constraints. 
    for i in range(num_nodes):
        for j in range(num_nodes):
            if capacity_matrix[i][j] > 0:
                edges.append((i,j))
                capacities.append(capacity_matrix[i][j])

    num_edges = len(edges)

    # C is the optimization function, in this case we want to maximize
    # the flow leaving the source which is the same as the flow reaching
    # the sink. -1 is used because linprog only minimizes.  
    c = np.zeros(num_edges)
    for i, (u,v) in enumerate(edges):
        if u == source:
            c[i] = -1

    # A_eq is the equality constraint matrix in this case that the 
    # flow into a node must be the same as the flow out to the node and 
    # that the flow through an edge cannot exceede the capacity of the edge
    # B_EQ is the equality constraint vector or the value the corresponding
    # A_EQ value must equal
    A_eq = []
    B_eq = []

    # This section fills out A_eq by determining if which nodes
    # have flow in and out and ensuring that he flow in and out is 
    # equivalent. For example at node 1 it will ensure that the 
    # value of the flow entering the nodes is the same as the flow
    # exiting by assigning a coefficient of either +1 for flows 
    # entering or -1 for flows exiting. B_eq always equals zero because
    # the flow in should be equivalent to he flow out which would equal 
    # zero.
    for node in range(num_nodes):
        if node == source or node == sink:
            continue
        constraint = np.zeros(num_edges)
        for i, (u,v) in enumerate(edges):
            if v == node:
                constraint[i] = 1
    
        for i, (u,v) in enumerate(edges):
            if u == node:
                constraint[i] = -1

        A_eq.append(constraint)
        B_eq.append(0)

    # The bound for each edge is 0 for the lower
    # and the capcity of the edge for the upper 
    # bound.
    bounds = [(0, cap) for cap in capacities]


    # running linprog to get the anser.
    result = linprog(c=c, A_eq=np.array(A_eq), b_eq=np.array(B_eq), bounds=bounds)

    if result.success:
        max_flow = -result.fun  # Negate because we minimized the negative flow
        print(f"Maximum Flow: {max_flow}")
    else:
        print("Optimization failed.")

def main():
    # Representing the graph as an adjacency matrix
    # using a 2D numpy array
    capacity_matrix1 = np.array([
    [0, 10, 0, 10, 0, 0],
    [0, 0, 3, 2, 2, 5], 
    [0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0]
    ])
    # Keeping track of the source, sink, and size
    source1 = 0
    sink1 = 5

    MaxFlow(capacity_matrix1, source1, sink1)
    
    # moved weight of edge AD to CD
    capacity_matrix2 = np.array([
    [0, 10, 0, 10, 0, 0],
    [0, 0, 3, 0, 2, 5], 
    [0, 0, 0, 0, 0, 5],
    [0, 0, 2, 0, 3, 0],
    [0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0]
    ])
    # Keeping track of the source, sink, and size
    source2 = 0
    sink2 = 5

    MaxFlow(capacity_matrix2, source2, sink2)

if __name__ == "__main__":
    main()
