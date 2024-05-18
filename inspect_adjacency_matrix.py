import pickle

# Load the adjacency matrix
with open('adjacency_matrix.pkl', 'rb') as f:
    adjacency_matrix = pickle.load(f)

# Print the shape of the adjacency matrix
print("Shape of adjacency matrix:", adjacency_matrix.shape)
