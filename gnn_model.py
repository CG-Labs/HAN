import spektral
from spektral.layers import GCNConv
from spektral.utils import normalized_laplacian, add_self_loops
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
import numpy as np
import pickle
import os
import sys
import json

class CustomGCNConv(GCNConv):
    def __init__(self, channels, units, **kwargs):
        super(CustomGCNConv, self).__init__(channels=channels, units=units, **kwargs)
        self.units = units

    def call(self, inputs, **kwargs):
        """
        Forward pass for the custom GCNConv layer.

        Parameters:
        - inputs: list, containing the feature matrix and adjacency matrix.

        Returns:
        - The output of the layer.
        """
        # Ignore the mask parameter by not passing it to the parent call
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Parameters:
        - input_shape: list of shapes, where the first element is the shape of the node features
                       and the second element is the shape of the adjacency matrix.

        Returns:
        - output_shape: list of shapes, where the first element is the output shape of the node features
                        (same spatial dimensions as input, but with the number of units as the feature dimension)
                        and the second element is the same shape as the input adjacency matrix.
        """
        feature_shape, adjacency_shape = input_shape
        output_feature_shape = (feature_shape[0], self.units)
        return [output_feature_shape, adjacency_shape]

@tf.keras.utils.register_keras_serializable()
class GNNModel(Model):
    def get_config(self):
        # Serialize the constructor parameters to a config dictionary
        config = super(GNNModel, self).get_config()
        config.update({
            'num_features': self.conv1.channels
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Deserialize the constructor parameters from the config dictionary
        num_features = config.get('num_features', None)
        if num_features is not None:
            return cls(num_features=num_features)
        else:
            raise ValueError("Missing required config parameter: num_features")

    def __init__(self, num_features, **kwargs):
        """
        Initialize the Graph Neural Network model with the given number of features.

        Parameters:
        - num_features: int, the number of features in the input data.
        """
        super(GNNModel, self).__init__(**kwargs)
        # Define the input shape based on the number of features
        self.input_shape = (None, num_features)
        # Initialize the convolutional layers with the correct number of channels and units
        self.conv1 = CustomGCNConv(channels=num_features, units=2, activation='relu', name='custom_gcn_conv')
        self.conv2 = CustomGCNConv(channels=16, units=16, activation='relu', name='custom_gcn_conv_1')
        self.dropout = Dropout(0.5, name='dropout')
        self.num_features = num_features
        # Initialize the dense layer with 1 unit for the output
        self.dense = Dense(units=1, activation='linear', name='dense')
        # Build the model with the defined input shape
        self.build((None, num_features))

    def build(self, input_shape):
        """
        Create the weights of the model's layers based on the input shape.

        Parameters:
        - input_shape: tuple, the shape of the input data.
        """
        # Ensure the input shape has a concrete feature dimension
        if input_shape is None or input_shape[1] is None:
            raise ValueError("Input shape must be a tuple with a concrete feature dimension.")
        input_shape_with_batch = (None, input_shape[1])  # Add the batch dimension back for compatibility
        # Ensure the input shape is not None before calling the build method
        if input_shape_with_batch[1] is None:
            raise ValueError("Feature dimension of input shape cannot be None.")

        # Ensure the input shape is a list with the correct shapes for node features and adjacency matrix
        input_shape_with_batch = [(None, input_shape[1]), (None, None)]  # Example adjacency shape, to be replaced with actual shape
        print("Input shape with batch before conv1 build:", input_shape_with_batch)  # Debug print
        self.conv1.build(input_shape_with_batch)
        print("Output shape after conv1 build:", self.conv1.compute_output_shape(input_shape_with_batch))  # Debug print
        input_shape_with_batch[0] = (None,) + tuple(self.conv1.compute_output_shape(input_shape_with_batch)[0][1:])
        print("Input shape with batch before conv2 build:", input_shape_with_batch)  # Debug print
        self.conv2.build(input_shape_with_batch)
        print("Output shape after conv2 build:", self.conv2.compute_output_shape(input_shape_with_batch))  # Debug print
        input_shape_with_batch[0] = (None,) + self.conv2.compute_output_shape(input_shape_with_batch)[0][1:]

        # Ensure the dense layer receives the correct input shape
        print("Input shape with batch before dense build:", input_shape_with_batch)  # Debug print
        # Flatten the input shape for the dense layer to match the output of the conv2 layer
        input_shape_with_batch[0] = (None, 16)
        self.dense.build(input_shape_with_batch[0])
        print("Output shape after dense build:", self.dense.compute_output_shape(input_shape_with_batch[0]))  # Debug print
        super(GNNModel, self).build(input_shape)  # Mark the model as built

    def call(self, inputs, training=False):
        """
        Forward pass for the model.

        Parameters:
        - inputs: tuple, containing the feature matrix and adjacency matrix.
        - training: bool, indicating whether the call is for training or inference.

        Returns:
        - The output of the last layer of the model.
        """
        x, adjacency = inputs
        x = self.conv1([x, adjacency])
        x = self.dropout(x, training=training)
        x = self.conv2([x, adjacency])
        # Flatten the output to match the expected input shape of the dense layer
        x = tf.reshape(x, (-1, self.conv2.units))
        # Pass the reshaped output to the dense layer for prediction
        prediction = self.dense(x)
        return prediction

    def analyze_data(self, processed_data):
        """
        Analyze the processed data using the GNN model to generate embeddings.

        Parameters:
        - processed_data: tuple, containing the feature matrix and adjacency matrix.

        Returns:
        - A dictionary with the embeddings generated by the model.
        """
        x, adjacency = processed_data
        embeddings = self.call((x, adjacency), training=False)
        return {'embeddings': embeddings.numpy()}

    def make_prediction(self, analysis_results):
        """
        Make predictions based on the analysis results using the GNN model.

        Parameters:
        - analysis_results: dict, containing the embeddings from the analyze_data method.

        Returns:
        - A dictionary with the prediction value.
        """
        embeddings = analysis_results['embeddings']
        # Reshape the embeddings to match the expected input shape of the dense layer
        embeddings_reshaped = tf.reshape(embeddings, (-1, self.conv2.units))
        # Using the dense layer to make a prediction from reshaped embeddings
        prediction = self.dense(embeddings_reshaped)
        return {'prediction': prediction.numpy()[0]}

    def train_model(self, feature_matrix, adjacency_matrix, labels, epochs=200, learning_rate=0.01):
        """
        Train the GNN model with the provided feature matrix and adjacency matrix.

        Parameters:
        - feature_matrix: ndarray, the feature matrix for the GNN.
        - adjacency_matrix: ndarray, the adjacency matrix for the GNN.
        - labels: ndarray, the labels for the training data.
        - epochs: int, the number of epochs to train the model.
        - learning_rate: float, the learning rate for the optimizer.

        Returns:
        - A dictionary indicating the status of the training process.
        """
        # Preprocess the adjacency matrix
        adjacency_matrix = spektral.utils.convolution.gcn_filter(adjacency_matrix)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = self.call((feature_matrix, adjacency_matrix), training=True)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            print(f"Epoch {epoch}: Loss: {loss.numpy()}")

        # Save the model architecture to a JSON file
        model_json = self.to_json()
        with open('trained_gnn_model.json', 'w') as json_file:
            json_file.write(model_json)
        # Save the weights to an HDF5 file
        self.save_weights('trained_gnn_model.weights.h5')

        return {'status': 'Model trained successfully', 'model_path': 'trained_gnn_model.json', 'weights_path': 'trained_gnn_model.weights.h5'}

    def load_trained_model(self, model_json_path, model_weights_path):
        """
        Load a trained GNN model from the specified JSON and HDF5 files.

        Parameters:
        - model_json_path: str, the path to the saved model JSON file.
        - model_weights_path: str, the path to the saved model weights HDF5 file.

        Returns:
        - The loaded GNN model.
        """
        # Load the model architecture from JSON file
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        model_config = json.loads(model_json)

        # Ensure custom layers are registered for deserialization
        custom_objects = {'GNNModel': GNNModel}

        # Reconstruct the model from the JSON file using the from_config method
        reconstructed_model = GNNModel.from_config(model_config['config'], custom_objects=custom_objects)

        # Build the model with the correct input shape and initialize weights
        dummy_feature_matrix = np.zeros((1, reconstructed_model.num_features))
        dummy_adjacency_matrix = np.zeros((1, 1))
        reconstructed_model.call((dummy_feature_matrix, dummy_adjacency_matrix), training=False)

        # Load the model weights from HDF5 file
        reconstructed_model.load_weights(model_weights_path)

        return reconstructed_model

    def predict_bitcoin_price(self, feature_matrix_path, adjacency_matrix_path, model_path, scaler_path):
        """
        Predict the Bitcoin price using the trained GNN model and the provided input data.

        Parameters:
        - feature_matrix_path: str, the path to the feature matrix pickle file.
        - adjacency_matrix_path: str, the path to the adjacency matrix pickle file.
        - model_path: str, the path to the saved trained model file (h5).
        - scaler_path: str, the path to the saved scaler file (pkl).

        Returns:
        - prediction: float, the predicted Bitcoin price.
        """
        import pickle

        # Load the trained model
        trained_model = self.load_trained_model('trained_gnn_model.json', 'trained_gnn_model.weights.h5')

        # Load the feature matrix
        try:
            with open(feature_matrix_path, 'rb') as f:
                feature_matrix = pickle.load(f)
        except FileNotFoundError:
            print(f"The feature matrix file {feature_matrix_path} was not found.")
            return "Feature matrix file not found."

        print("Loaded feature matrix shape:", feature_matrix.shape)

        # Load the adjacency matrix
        try:
            with open(adjacency_matrix_path, 'rb') as f:
                adjacency_matrix = pickle.load(f)
        except FileNotFoundError:
            print(f"The adjacency matrix file {adjacency_matrix_path} was not found.")
            return "Adjacency matrix file not found."

        # Make prediction
        analysis_results = trained_model.analyze_data((feature_matrix, adjacency_matrix))
        prediction_result = trained_model.make_prediction(analysis_results)
        prediction = prediction_result['prediction']

        return prediction

if __name__ == "__main__":
    # Default to prediction mode if no argument is provided
    mode = 'predict' if len(sys.argv) == 1 else sys.argv[1]

    if mode == 'predict':
        # Paths to the necessary files
        input_data_path = 'input_data_2024.csv'  # Path to the input data for end of 2024
        model_path = 'trained_gnn_model.h5'
        scaler_path = 'scaler.pkl'  # Path to the scaler used during training

        # Load the feature matrix to determine the number of features
        try:
            with open('feature_matrix.pkl', 'rb') as f:
                feature_matrix = pickle.load(f)
            num_features = feature_matrix.shape[1]
        except FileNotFoundError:
            print("Feature matrix file not found.")
            exit(1)

        # Initialize the GNN model for prediction
        gnn_model = GNNModel(num_features=num_features)

        # Load the trained model architecture and weights
        gnn_model.load_trained_model('trained_gnn_model.json', 'trained_gnn_model.weights.h5')

        # Make a prediction
        prediction = gnn_model.predict_bitcoin_price('feature_matrix.pkl', 'adjacency_matrix.pkl', 'trained_gnn_model.json', scaler_path)
        print(f"Predicted Bitcoin price at the end of 2024: {prediction}")
    # else:
    #     # Training mode is disabled to focus on prediction
    #     print("Training mode is not enabled. Exiting.")
    #     exit(0)
