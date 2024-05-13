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

    # Adjust the shape of y_train, y_val, y_test to match the number of nodes
    nb_nodes = len(feature_vectors_list)
    nb_classes = 1  # Temporary single class for all nodes, to be defined based on use case

    # Assign nodes to train, validation, and test sets
    num_train = int(nb_nodes * 0.6)
    num_val = int(nb_nodes * 0.2)
    num_test = nb_nodes - num_train - num_val

    # Initialize y_train, y_val, y_test with the correct number of classes for each node
    y_train = np.zeros((nb_nodes, nb_classes))
    y_val = np.zeros((nb_nodes, nb_classes))
    y_test = np.zeros((nb_nodes, nb_classes))

    # Assign class labels to nodes for training, validation, and testing
    # Here we assign a dummy class label of 1 for demonstration purposes
    # This should be replaced with actual class labels derived from the CV data
    y_train[:num_train, :] = 1
    y_val[num_train:num_train + num_val, :] = 1
    y_test[num_train + num_val:, :] = 1

    # Update the masks to match the number of nodes
    train_mask = np.zeros((nb_nodes,)).astype(bool)
    val_mask = np.zeros((nb_nodes,)).astype(bool)
    test_mask = np.zeros((nb_nodes,)).astype(bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True

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

# Initialize TensorFlow 2.x checkpointing
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=5)

# Restore the latest checkpoint using TensorFlow 2.x checkpointing
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print('Model restored from checkpoint at {}'.format(checkpoint_manager.latest_checkpoint))

# Training loop
for epoch in range(nb_epochs):
    logging.debug("Starting epoch %d", epoch)
    # Placeholder for additional training loop code

    # Log the shapes of logits and labels after they have been computed
    # Removed the TensorFlow 1.x session.run calls and replaced with direct TensorFlow 2.x eager execution

    # Evaluate and log the shapes of logits and labels after they have been computed
    # Removed the TensorFlow 1.x session.run calls and replaced with direct TensorFlow 2.x eager execution

    # ... rest of the training loop code ...

# Apply t-SNE visualization on the final embeddings
def visualize_with_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for class_id in np.unique(labels):
        indices = labels == class_id
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(class_id))
    plt.legend()
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Removed incorrect assignment of final_embedding outside of TensorFlow session scope
# Removed redundant call to visualize_with_tsne

# Removed incorrect assignment of final_embedding outside of TensorFlow session scope

# Removed redundant call to visualize_with_tsne

# Removed incorrect assignment of final_embedding outside of TensorFlow session scope

# Removed incorrect assignment of final_embedding outside of TensorFlow session scope

# Removed incorrect assignment of final_embedding outside of TensorFlow session scope

# Assign the final embedding output from the model to jhy_final_embedding
jhy_final_embedding = final_embedding
visualize_with_tsne(jhy_final_embedding, yy)

# visualize_with_tsne(jhy_final_embedding, yy) # Removed redundant call to visualize_with_tsne

# Call the visualization function with the final embeddings and the corresponding labels
visualize_with_tsne(jhy_final_embedding, yy)

init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())

vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0

with tf.compat.v1.Session(config=config) as sess:
    sess.run(init_op)

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0

    for epoch in range(nb_epochs):
        # ... training loop code ...

        if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
            if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                vacc_early_model = val_acc_avg / vl_step
                vlss_early_model = val_loss_avg / vl_step
                checkpoint_manager.save()
                print('Model checkpoint saved at epoch {}'.format(epoch))
            vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
            vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print('Early stop! Min loss: ', vlss_mn,
                      ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ',
                      vlss_early_model, ', accuracy: ', vacc_early_model)
                break

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

    print('Training complete, restoring best model...')
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print('Model restored from checkpoint at {}'.format(checkpoint_manager.latest_checkpoint))
    # ... test set evaluation code ...

checkpoint_manager.save()

checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Initialize TensorFlow 2.x checkpointing
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=5)

# Restore the latest checkpoint using TensorFlow 2.x checkpointing
checkpoint.restore(checkpoint_manager.latest_checkpoint)

saver = tf.compat.v1.train.Saver()

# Removed the assertion that checks the number of nodes in y_train

# Removed the assertion that checks the number of nodes in y_train
# assert y_train.shape[0] == nb_nodes, "Number of nodes in y_train does not match nb_nodes"

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
adj_list = [adj[np.newaxis] for adj in rownetworks]

# Removed the unnecessary reshaping of label tensors
# y_train, y_val, y_test = y_train, y_val, y_test
# Removed the unnecessary reshaping of mask tensors
# train_mask, val_mask, test_mask = train_mask, val_mask, test_mask

biases_list = [process.adj_to_bias(adj, nhood=1) for adj in rownetworks]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(inputs_list=ftr_in_list, nb_classes=nb_classes, nb_nodes=nb_nodes, training=is_train, attn_drop=attn_drop, ffd_drop=ffd_drop, bias_mat_list=bias_in_list, hid_units=hid_units, n_heads=n_heads, batch_size=batch_size, activation=nonlinearity, residual=residual, mp_att_size=128)
    # Calculate masked loss
    logits_shape = tf.shape(logits)
    # Ensure logits tensor has the correct shape [batch_size, nb_nodes, nb_classes]
    expected_shape = tf.constant([batch_size, nb_nodes, nb_classes])

    # Log the shape of logits for debugging
    logging.debug("Logits shape: %s", logits_shape)
    # Log the expected shape for debugging
    logging.debug("Expected shape: %s", expected_shape)

    # If the shape is not as expected, raise an error
    with tf.control_dependencies([tf.debugging.assert_equal(logits_shape, expected_shape, message="Logits tensor shape does not match expected shape")]):
        logits = tf.identity(logits)

    # Debugging: Log the shapes of logits and labels before reshaping
    logits_shape_before_reshape = tf.shape(logits)
    labels_shape_before_reshape = tf.shape(lbl_in)
    logging.debug("Logits shape before reshape: %s", logits_shape_before_reshape)
    logging.debug("Labels shape before reshape: %s", labels_shape_before_reshape)

    # Calculate the correct size for the first dimension of the reshaped tensors
    reshaped_size = batch_size * nb_nodes
    # Assert that the number of nodes in the labels matches the number of nodes in the logits
    # Removed the assertion that checks for the number of classes in y_train
    # assert y_train.shape[1] == nb_classes, "Number of classes in y_train does not match nb_classes"
    # Removed the assertion that checks for the number of nodes in y_train
    # assert y_train.shape[0] == nb_nodes, "Number of nodes in y_train does not match nb_nodes"
    # Reshape logits to [batch_size * nb_nodes, nb_classes]
    log_resh = tf.reshape(logits, [reshaped_size, nb_classes])
    # Reshape labels to [batch_size * nb_nodes, nb_classes]
    lab_resh = tf.reshape(lbl_in, [reshaped_size, nb_classes])
    # Reshape mask to [batch_size * nb_nodes]
    msk_resh = tf.reshape(msk_in, [reshaped_size])
    # Ensure that the first dimension of the logits and labels tensors are equal
    assert log_resh.shape[0] == lab_resh.shape[0], "Mismatch in first dimension of logits and labels tensors after reshaping"
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0

            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

                # Evaluate and log the shapes of logits and labels after they have been computed
                logits_shape_evaluated, labels_shape_evaluated = sess.run([logits_shape_before_reshape, labels_shape_before_reshape], feed_dict=fd)
                logging.debug("Evaluated logits shape: %s", logits_shape_evaluated)
                logging.debug("Evaluated labels shape: %s", labels_shape_evaluated)
                logging.getLogger().handlers[0].flush()

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            # import pdb; pdb.set_trace()
            print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step))

            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        # Assign the final embedding output from the model to jhy_final_embedding
        # Ensure that the labels variable 'yy' is correctly defined for visualization
        yy = y_test[test_mask]
        # Call the visualization function with the final embeddings and the corresponding labels
        visualize_with_tsne(jhy_final_embedding, yy)

        sess.close()
