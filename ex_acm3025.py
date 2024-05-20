import time
import numpy as np
import tensorflow as tf
import yfinance as yf

from models import GAT, HeteGAT, HeteGAT_multi
from utils import process

# 禁用gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

dataset = 'acm'
featype = 'fea'
print('Loading model checkpoint file...')
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 5  # Reduced number of epochs for quicker testing
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

def fetch_bitcoin_data(start_date, end_date):
    """
    Fetch historical Bitcoin data from Yahoo Finance.
    """
    print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
    bitcoin_data = yf.download('BTC-USD', start=start_date, end=end_date)
    print("Data fetching complete.")
    print("Bitcoin data shape:", bitcoin_data.shape)
    return bitcoin_data

def preprocess_data(data):
    """
    Preprocess the Bitcoin data for the model.
    Normalize the 'Close' prices and reshape the data for the model input.
    """
    print("Starting data preprocessing...")
    # Normalize the 'Close' prices
    close_prices = data['Close'].values
    normalized_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)

    # Reshape the data to match the model's input shape
    # Assuming the model expects a 3D input shape (batch_size, timesteps, features)
    # Here we use a sliding window approach to create a sequence of prices for each prediction
    timesteps = 10  # The number of timesteps the model looks back for making a prediction
    features = 1    # The number of features used for prediction, here it's just the normalized 'Close' price
    samples = len(normalized_prices) - timesteps + 1

    X = np.zeros((samples, timesteps, features))
    for i in range(samples):
        X[i] = normalized_prices[i:i+timesteps].reshape(timesteps, features)

    print("Data preprocessing complete. Shape of preprocessed data:", X.shape)
    return X

# Fetch and preprocess Bitcoin data
start_date = '2020-01-01'  # Start date for fetching historical data
print('Starting data fetching process...')
end_date = '2023-05-11'    # End date for fetching historical data, set to the current date
bitcoin_data = fetch_bitcoin_data(start_date, end_date)
preprocessed_data = preprocess_data(bitcoin_data)

# Define the number of timesteps and features for the model input
timesteps = 10  # The number of timesteps the model looks back for making a prediction
features = 1    # The number of features used for prediction, here it's just the normalized 'Close' price

ftr_in = tf.keras.Input(shape=(timesteps, features), name='ftr_in')

# Correct the number of nodes to match the actual graph size for the bias matrix
nb_nodes = 496  # This should be set to the actual number of nodes in the graph
adjacency_matrix = np.eye(nb_nodes, k=1) + np.eye(nb_nodes, k=-1)
adjacency_matrix = np.array([adjacency_matrix])  # Add batch dimension

# Compute the bias matrix using the adjacency matrix with the correct shape
bias_mat = process.adj_to_bias(adjacency_matrix, [nb_nodes], nhood=1)

# Apply the attention mechanism and capture the tensor shapes using PrintShapeLayer
logits, shapes = GAT.inference(ftr_in, nb_classes=1, nb_nodes=preprocessed_data.shape[1], training=False,
                               attn_drop=0.6, ffd_drop=0.6, bias_mat=bias_mat, hid_units=hid_units,
                               n_heads=n_heads, activation=tf.nn.elu, residual=False, return_shapes=True)

# Print the captured tensor shapes
print("Tensor shapes:", shapes)

# Create a Keras model
model = tf.keras.Model(inputs=ftr_in, outputs=logits)

# Run the model to get the predictions
predictions = model.predict(preprocessed_data)

# Print the actual shapes of the tensors
print("Actual tensor shapes:", predictions.shape)

import scipy.sparse as sp
# Generate dummy labels and masks based on the preprocessed data shape
num_samples = preprocessed_data.shape[0]
num_classes = 2  # Assuming binary classification for simplicity
y_train = np.random.randint(0, 2, size=(num_samples, num_classes))
y_val = np.random.randint(0, 2, size=(num_samples, num_classes))
y_test = np.random.randint(0, 2, size=(num_samples, num_classes))
train_mask = np.random.choice([True, False], size=(num_samples,))
val_mask = np.random.choice([True, False], size=(num_samples,))
test_mask = np.random.choice([True, False], size=(num_samples,))

print(f"Generated dummy labels and masks for training, validation, and testing.")

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

# Adjust the bias matrix to have the correct shape [1, nb_nodes, nb_nodes]
biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, feature_list, bias_list, labels, mask, batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.feature_list = feature_list
        self.bias_list = bias_list
        self.mask = mask
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.feature_list[0]))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(len(self.feature_list[0]) / self.batch_size))
        print(f"Number of batches per epoch: {num_batches}")
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Check if the batch is empty
        if len(indexes) == 0:
            print(f"Warning: Empty batch detected at index {index}. Skipping batch.")
            return None

        # Find selected samples
        feature_list_temp = [np.squeeze(self.feature_list[k][indexes], axis=0) for k in range(len(self.feature_list))]
        bias_list_temp = [np.squeeze(self.bias_list[k][indexes], axis=0) for k in range(len(self.bias_list))]

        # Generate data
        X, y, mask = self.__data_generation(feature_list_temp, bias_list_temp, indexes)

        print(f"Generated batch {index + 1} with {len(indexes)} samples. Batch indexes: {indexes}")
        return X, y, mask

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, feature_list_temp, bias_list_temp, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = [sp.hstack(feature_list_temp, format='csr')]
        y = self.labels[indexes]
        mask = self.mask[indexes]

        return X, y, mask

# Instantiate the model and call the inference method
print("Instantiating the model...")
logits, final_embed, att_val = HeteGAT_multi.inference(inputs_list=fea_list, nb_classes=nb_classes, nb_nodes=nb_nodes, training=False, attn_drop=0.6, ffd_drop=0.6, bias_mat_list=biases_list, hid_units=hid_units, n_heads=n_heads, activation=tf.nn.elu, residual=residual, mp_att_size=128)
print("Model instantiated with logits shape:", logits.shape, "and final_embed shape:", final_embed.shape)

# Create a Keras model using the logits returned by the inference method
model = tf.keras.Model(inputs=fea_list, outputs=logits)
print("Keras model created.")

# Compile the Keras model with optimizer and loss function
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Model compiled.")

# Prepare the data generators for training and validation
print("Preparing data generators...")
train_data = DataGenerator(fea_list, biases_list, y_train, train_mask, batch_size)
val_data = DataGenerator(fea_list, biases_list, y_val, val_mask, batch_size)
print("Data generators ready.")

# Train the model using the fit method
print("Starting model training at", time.strftime("%Y-%m-%d %H:%M:%S"))
history = model.fit(train_data, validation_data=val_data, epochs=nb_epochs)
print("Model training completed at", time.strftime("%Y-%m-%d %H:%M:%S"))

# Save the trained model
print("Saving the trained model...")
model.save('trained_model.h5')
print("Trained model saved at 'trained_model.h5'. Model summary:")
model.summary()

# Load the model for evaluation
print("Loading the model for evaluation...")
model = tf.keras.models.load_model('trained_model.h5')
print("Model loaded for evaluation. Model summary:")
model.summary()

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_data = DataGenerator(fea_list, biases_list, y_test, test_mask, batch_size)
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test evaluation complete. Loss: {test_loss}; Accuracy: {test_accuracy}")
