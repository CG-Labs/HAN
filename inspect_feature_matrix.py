import pickle

# Load the feature matrix
with open('feature_matrix.pkl', 'rb') as f:
    feature_matrix = pickle.load(f)

# Print the shape of the feature matrix
print("Shape of feature matrix:", feature_matrix.shape)
