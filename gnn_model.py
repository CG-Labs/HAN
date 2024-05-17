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

@tf.keras.utils.register_keras_serializable()
class GNNModel(Model):
    def get_config(self):
        # Serialize the constructor parameters to a config dictionary
        config = super(GNNModel, self).get_config()
        config.update({
            'num_classes': self.conv2.units,
            'num_features': self.num_features
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the constructor parameters from the config dictionary
        num_classes = config.pop('num_classes')
        num_features = config.pop('num_features')
        return cls(num_classes=num_classes, num_features=num_features, **config)

    def __init__(self, num_classes, num_features, **kwargs):
        """
        Initialize the Graph Neural Network model with the given number of classes and features.

        Parameters:
        - num_classes: int, the number of classes for classification tasks.
        - num_features: int, the number of features in the input data.
        """
        super(GNNModel, self).__init__(**kwargs)
        self.conv1 = CustomGCNConv(channels=num_features, units=16, activation='relu')
        self.conv2 = CustomGCNConv(channels=16, units=num_classes, activation='linear')
        self.dropout = Dropout(0.5)
        self.num_features = num_features
        # Adding a dense layer for regression prediction
        self.dense = Dense(1, activation='linear')

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
        # Using the dense layer to output a single value for regression
        x = self.dense(x)
        return x

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
        # Using the dense layer to make a prediction from embeddings
        prediction = self.dense(embeddings)
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
        self.save('trained_gnn_model.h5')
        return {'status': 'Model trained successfully', 'model_path': 'trained_gnn_model.h5'}

    def load_trained_model(self, model_path):
        """
        Load a trained GNN model from the specified path.

        Parameters:
        - model_path: str, the path to the saved model file.

        Returns:
        - The loaded GNN model.
        """
        with tf.keras.utils.custom_object_scope({'CustomGCNConv': CustomGCNConv, 'GNNModel': GNNModel}):
            return tf.keras.models.load_model(model_path)

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
        trained_model = self.load_trained_model(model_path)

        # Load the feature matrix
        try:
            with open(feature_matrix_path, 'rb') as f:
                feature_matrix = pickle.load(f)
        except FileNotFoundError:
            print(f"The feature matrix file {feature_matrix_path} was not found.")
            return "Feature matrix file not found."

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
    # Default to training mode if no argument is provided
    mode = 'train' if len(sys.argv) == 1 else sys.argv[1]

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
        gnn_model = GNNModel(num_classes=1, num_features=num_features)

        # Make a prediction
        prediction = gnn_model.predict_bitcoin_price('feature_matrix.pkl', 'adjacency_matrix.pkl', model_path, scaler_path)
        print(f"Predicted Bitcoin price at the end of 2024: {prediction}")
    else:
        # Training mode
        # Load the feature matrix and adjacency matrix
        try:
            with open('feature_matrix.pkl', 'rb') as f:
                feature_matrix = pickle.load(f)
        except FileNotFoundError:
            print("Feature matrix file not found.")
            exit(1)

        try:
            with open('adjacency_matrix.pkl', 'rb') as f:
                adjacency_matrix = pickle.load(f)
        except FileNotFoundError:
            print("Adjacency matrix file not found.")
            exit(1)

        # Load the labels for the training data
        try:
            with open('labels.pkl', 'rb') as f:
                labels = pickle.load(f)
        except FileNotFoundError:
            print("Labels file not found.")
            exit(1)

        # Initialize the GNN model
        num_classes = 1  # For regression, we have one output
        num_features = feature_matrix.shape[1]
        gnn_model = GNNModel(num_classes=num_classes, num_features=num_features)

        # Train the model
        training_status = gnn_model.train_model(feature_matrix, adjacency_matrix, labels)
        print(training_status['status'])
