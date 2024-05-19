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

# Define the number of timesteps and features for the model input
timesteps = 10  # The number of timesteps the model looks back for making a prediction
features = 1    # The number of features used for prediction, here it's just the normalized 'Close' price

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
        return int(np.floor(len(self.feature_list[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find selected samples
        feature_list_temp = [np.squeeze(self.feature_list[k][indexes], axis=0) for k in range(len(self.feature_list))]
        bias_list_temp = [np.squeeze(self.bias_list[k][indexes], axis=0) for k in range(len(self.bias_list))]

        # Generate data
        X, y, mask = self.__data_generation(feature_list_temp, bias_list_temp, indexes)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

# xx = xx / LA.norm(xx, axis=1)
yy = y_test[test_mask]

print('xx: {}, yy: {}'.format(xx.shape, yy.shape))

def my_KNN(embeddings, labels, n_neighbors=5):
    """
    Perform K-Nearest Neighbors classification on the embeddings.
    Args:
        embeddings: The embeddings generated by the model.
        labels: The true labels for the embeddings.
        n_neighbors: The number of neighbors to use for KNN.
    Returns:
        The classification accuracy.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(embeddings, labels)
    pred_labels = knn.predict(embeddings)
    accuracy = accuracy_score(labels, pred_labels)
    print(f'KNN classification accuracy: {accuracy}')
    return accuracy

def my_Kmeans(embeddings, labels, n_clusters=5):
    """
    Perform K-Means clustering on the embeddings and compare with true labels.
    Args:
        embeddings: The embeddings generated by the model.
        labels: The true labels for the embeddings.
        n_clusters: The number of clusters to form.
    Returns:
        The cluster labels and the adjusted Rand index comparing the clusters with true labels.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f'Adjusted Rand index: {ari}')
    return cluster_labels, ari

my_KNN(xx, yy)
my_Kmeans(xx, yy)
