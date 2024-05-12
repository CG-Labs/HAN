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

nb_classes = 3  # Assuming 3 classes for the purpose of generating dummy data

# jhy data
import scipy.io as sio
import scipy.sparse as sp


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data_dblp(path='structured_cv_data.txt'):
    # Load structured CV data from text file
    cv_data = process_cv_data(path)

    # Use the processed CV data to create feature vectors and adjacency matrices
    rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = cv_data

    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~

adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()
if featype == 'adj':
    fea_list = adj_list



import scipy.sparse as sp




nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]

# Removed the redefinition of nb_classes to maintain the correct predefined value

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]

# Removed the unnecessary reshaping of label tensors
# y_train, y_val, y_test = y_train, y_val, y_test
# Removed the unnecessary reshaping of mask tensors
# train_mask, val_mask, test_mask = train_mask, val_mask, test_mask

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

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
    assert y_train.shape[1] == nb_classes, "Number of classes in y_train does not match nb_classes"
    assert y_train.shape[0] == nb_nodes, "Number of nodes in y_train does not match nb_nodes"
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
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
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

        print('start knn, kmean.....')
        xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

        from numpy import linalg as LA

        # xx = xx / LA.norm(xx, axis=1)
        yy = y_test[test_mask]

        print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        from jhyexps import my_KNN, my_Kmeans#, my_TSNE, my_Linear

        my_KNN(xx, yy)
        my_Kmeans(xx, yy)

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

        # Call the visualization function with the final embeddings and the corresponding labels
        visualize_with_tsne(jhy_final_embedding, yy)

        sess.close()
