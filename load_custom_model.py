import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from gnn_model import GNNModel
import pickle

# Load the feature matrix to determine the number of features
try:
    with open('feature_matrix.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    num_features = feature_matrix.shape[1]
except FileNotFoundError:
    print("Feature matrix file not found.")
    exit(1)

# Instantiate the GNNModel directly with the correct number of features
model = GNNModel(num_features=num_features)

# Load the weights manually from the HDF5 file
hdf5_path = 'trained_gnn_model.h5'
with h5py.File(hdf5_path, 'r') as f:
    for layer in model.layers:
        if layer.name in f['model_weights']:
            layer_group = f['model_weights'][layer.name]
            # Initialize lists to hold the kernel and bias weights separately
            kernel_weights = []
            bias_weights = []
            # Iterate over the weight names in the group and append the actual weights to the lists
            for weight_name in layer_group:
                # Ensure the item is a dataset before adding to weights
                if isinstance(layer_group[weight_name], h5py.Dataset):
                    # Retrieve the dataset
                    weight_dataset = layer_group[weight_name]
                    # Append the dataset value to the appropriate list
                    if 'kernel' in weight_name:
                        kernel_weights.append(np.array(weight_dataset))
                    elif 'bias' in weight_name:
                        bias_weights.append(np.array(weight_dataset))
                elif isinstance(layer_group[weight_name], h5py.Group):
                    # If the item is a group, iterate over its datasets
                    for sub_weight_name, sub_weight in layer_group[weight_name].items():
                        if isinstance(sub_weight, h5py.Dataset):
                            # Append the dataset value to the appropriate list
                            if 'kernel' in sub_weight_name:
                                kernel_weights.append(np.array(sub_weight))
                            elif 'bias' in sub_weight_name:
                                bias_weights.append(np.array(sub_weight))
            # Combine the kernel and bias weights
            weights = kernel_weights + bias_weights
            # Check if the weights lists are not empty before setting the weights
            if kernel_weights and bias_weights:
                # Set the weights to the layer
                try:
                    layer.set_weights(weights)
                except ValueError as e:
                    print(f"Error setting weights for layer {layer.name}: {e}")
                    # If there is a shape mismatch error, print the expected and actual shapes
                    if 'shape' in str(e):
                        expected_kernel_shape = layer.kernel.shape
                        expected_bias_shape = layer.bias.shape
                        provided_kernel_shape = kernel_weights[0].shape if kernel_weights else None
                        provided_bias_shape = bias_weights[0].shape if bias_weights else None
                        print(f"Expected kernel shape: {expected_kernel_shape}, provided kernel shape: {provided_kernel_shape}")
                        print(f"Expected bias shape: {expected_bias_shape}, provided bias shape: {provided_bias_shape}")
                        # Check if the provided shapes match the expected shapes
                        if provided_kernel_shape == expected_kernel_shape and provided_bias_shape == expected_bias_shape:
                            # Set the weights to the layer
                            layer.set_weights([kernel_weights[0], bias_weights[0]])
                        else:
                            print(f"Cannot set weights: shape mismatch for layer {layer.name}")
                else:
                    print(f"No weights found for layer {layer.name}")
        else:
            print(f"Layer {layer.name} not found in the HDF5 file")

# Print the model summary to verify the architecture
model.summary()
