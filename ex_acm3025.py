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
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
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
    bitcoin_data = yf.download('BTC-USD', start=start_date, end=end_date)
    return bitcoin_data

def preprocess_data(data):
    """
    Preprocess the Bitcoin data for the model.
    Normalize the 'Close' prices and reshape the data for the model input.
    """
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

    return X

# Fetch and preprocess Bitcoin data
start_date = '2023-01-01'  # Start date for fetching historical data
end_date = '2024-12-31'    # End date for fetching historical data
bitcoin_data = fetch_bitcoin_data(start_date, end_date)
preprocessed_data = preprocess_data(bitcoin_data)

# Model prediction code
# Define the model input layer using tf.keras.Input
ftr_in = tf.keras.Input(shape=(timesteps, features), name='ftr_in')

# Define the model using the functional API
gat_model = GAT(hid_units=hid_units, n_heads=n_heads, nb_classes=1, nb_nodes=preprocessed_data.shape[1],
                attn_drop=0.6, ffd_drop=0.6, activation=tf.nn.elu, residual=False)

# Assuming the model has a 'call' method to make predictions
predictions = gat_model(ftr_in, training=False)

# Create a Keras model
model = tf.keras.Model(inputs=ftr_in, outputs=predictions)

# Run the model to get the predictions
predictions = model.predict(preprocessed_data)

# The predictions variable now contains the Bitcoin price predictions

import scipy.sparse as sp

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

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')

# Removed TensorFlow 1.x ConfigProto and Session code
# Refactored to use TensorFlow 2.x and Keras API for model training and evaluation

# Define the model using the functional API
model = HeteGAT_multi(nb_classes=nb_classes, nb_nodes=nb_nodes, hid_units=hid_units, n_heads=n_heads,
                      attn_drop=0.6, ffd_drop=0.6, activation=tf.nn.elu, residual=residual)

# Compile the model with optimizer and loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              weighted_metrics=['accuracy'])

# Prepare the data generators for training and validation
train_data = DataGenerator(fea_list, biases_list, y_train, train_mask, batch_size)
val_data = DataGenerator(fea_list, biases_list, y_val, val_mask, batch_size)

# Train the model using the fit method
history = model.fit(train_data, validation_data=val_data, epochs=nb_epochs)

# Save the trained model
model.save('trained_model.h5')

# Load the model for evaluation
model = tf.keras.models.load_model('trained_model.h5')

# Evaluate the model on the test set
test_data = DataGenerator(fea_list, biases_list, y_test, test_mask, batch_size)
test_loss, test_accuracy = model.evaluate(test_data)

print('Test loss:', test_loss, '; Test accuracy:', test_accuracy)

print('start knn, kmean.....')
xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

from numpy import linalg as LA

# xx = xx / LA.norm(xx, axis=1)
yy = y_test[test_mask]

print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
from jhyexps import my_KNN, my_Kmeans#, my_TSNE, my_Linear

my_KNN(xx, yy)
my_Kmeans(xx, yy)
