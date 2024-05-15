import time
import numpy as np
import tensorflow as tf
import sys
import logging

from models import GAT, HeteGAT, HeteGAT_multi
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import process
from process_cv_data import process_cv_data

# 禁用gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Set up logging at the start to capture all messages
logging.basicConfig(filename='debug.log', level=logging.DEBUG, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

print("TensorFlow version:", tf.__version__)

dataset = 'acm'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 6  # Adjusted to match the first dimension of feature_vectors_tensor after concatenation
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
nb_classes = 3  # Assuming 3 classes for the purpose of generating dummy data

# Placeholder value for the number of graph nodes, to be updated with actual data

# The model instantiation is moved to after the load_data_dblp function call

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
# Removed the premature print statement for 'model' to prevent NameError

# jhy data
import scipy.io as sio
import scipy.sparse as sp

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_data_dblp(cv_path='Alan_Woulfe_CV.txt'):
    # Read the CV text file and process the CV text
    with open(cv_path, 'r') as file:
        cv_text = file.read()
    cv_data = process_cv_data(cv_text)

    # Unpack the processed CV data to create feature vectors and adjacency matrices
    feature_vectors_list, adjacency_matrix, y_train, y_val, y_test, train_mask, val_mask, test_mask = cv_data

    # The number of nodes is the length of the feature vectors list
    nb_nodes = len(feature_vectors_list)
    # The number of classes is the shape of the second dimension of y_train
    nb_classes = y_train.shape[1]

    # Instantiate the model with the correct number of nodes and classes
    model = HeteGAT_multi(nb_classes=nb_classes, nb_nodes=nb_nodes, hid_units=hid_units, n_heads=n_heads, activation=nonlinearity, residual=residual)

    return adjacency_matrix, feature_vectors_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, model

# Load data and ensure feature_vectors_list is a list and not empty before proceeding
rownetworks, feature_vectors_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, model = load_data_dblp()
logging.debug("Type of rownetworks: %s", type(rownetworks))
logging.debug("Contents of rownetworks: %s", rownetworks)
# Removed type and empty list checks for feature_vectors_list
# Use the feature vectors as they are, since they are already in the correct format
fea_list = feature_vectors_list
# Additional logging to confirm the structure of feature_vectors_list
logging.debug("Type of feature_vectors_list: %s", type(feature_vectors_list))
if isinstance(feature_vectors_list, list) and feature_vectors_list:
    logging.debug("First element of feature_vectors_list: %s", feature_vectors_list[0])
else:
    logging.error("feature_vectors_list is not a list or is empty")

if featype == 'adj':
    fea_list = adj_list

import scipy.sparse as sp

# Check if fea_list is not empty before accessing
if fea_list:
    # Log the type and content of the first element in fea_list for debugging
    logging.debug("Type of fea_list[0]: %s", type(fea_list[0]))
    logging.debug("Content of fea_list[0]: %s", fea_list[0])
    nb_nodes = fea_list[0].shape[0]
    ft_size = fea_list[0].shape[1]
else:
    logging.error("Feature list is empty. Cannot proceed with model training.")
    sys.exit("Error: Feature list is empty.")

# Initialize the checkpoint manager for saving and loading model checkpoints
checkpoint_prefix = './checkpoints/ckpt'

# Define the optimizer and loss function for the model training
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Reshape each feature vector to include the batch and node dimensions and concatenate along the batch dimension
feature_vectors_tensor = tf.concat([tf.reshape(fv, (1, nb_nodes, ft_size)) for fv in feature_vectors_list], axis=0)
logging.debug("Shape of feature_vectors_tensor after concatenation: %s", feature_vectors_tensor.shape)

# Ensure the feature_vectors_tensor is 3-dimensional and has the correct shape
assert len(feature_vectors_tensor.shape) == 3, "feature_vectors_tensor must be 3-dimensional"
assert feature_vectors_tensor.shape[0] == batch_size, "The first dimension of feature_vectors_tensor must match batch_size"
assert feature_vectors_tensor.shape[1] == nb_nodes, "The second dimension of feature_vectors_tensor must match the number of nodes"
assert feature_vectors_tensor.shape[2] == ft_size, "The third dimension of feature_vectors_tensor must match the feature size"

# Reshape y_train to match the batch size dimension of feature_vectors_tensor
y_train = np.expand_dims(y_train, axis=0)
logging.debug("Shape of y_train after reshaping: %s", y_train.shape)

# Ensure y_train is 3-dimensional and has the correct shape
assert len(y_train.shape) == 3, "y_train must be 3-dimensional"
assert y_train.shape[0] == batch_size, "The first dimension of y_train must match batch_size"
assert y_train.shape[1] == nb_nodes, "The second dimension of y_train must match the number of nodes"
assert y_train.shape[2] == nb_classes, "The third dimension of y_train must match the number of classes"

# Create a TensorFlow dataset with the correctly shaped tensors
train_dataset = tf.data.Dataset.from_tensor_slices((feature_vectors_tensor, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=5)

if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print('Model restored from checkpoint at {}'.format(checkpoint_manager.latest_checkpoint))

# Initialize metrics to track the loss and accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

# Initialize lists to collect embeddings and labels from each batch
all_embeddings = []
all_labels = []

# Generate bias matrices for each graph and log their shapes for verification
biases_list = []
for adj in rownetworks:
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    biases = tf.SparseTensor(indices=np.array([adj_normalized.row, adj_normalized.col]).T,
                             values=adj_normalized.data,
                             dense_shape=adj_normalized.shape)
    biases = tf.sparse.reorder(biases)  # Reorder the SparseTensor to sort the indices
    biases_list.append(tf.sparse.to_dense(biases))
logging.debug("Biases list populated with %d bias matrices", len(biases_list))

# Reset the metrics at the start of the next epoch
train_loss.reset_state()
train_accuracy.reset_state()

# Log every 200 batches.
# Moved inside the for loop to ensure 'step' is defined
# Log every 200 batches.
# Save the model every 5 epochs
# Training loop
for epoch in range(nb_epochs):
    start_time = time.time()
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_accuracy.reset_state()

    # Iterate over the batches of the dataset.
    for step, (batch_features, batch_labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Debug print to confirm the shape of batch_features before passing to the model
            tf.print("Debug: Shape of batch_features before model call:", tf.shape(batch_features))
            logits, _, _ = model(batch_features, biases_list, attn_drop=0.5, ffd_drop=0.5, training=True)
            # Compute loss
            loss_value = loss_fn(batch_labels, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metrics
        train_loss.update_state(loss_value)
        train_accuracy.update_state(batch_labels, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch, step, train_loss.result(), train_accuracy.result()))

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_manager.save()

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))

# End of script
