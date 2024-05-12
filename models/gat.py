import numpy as np
import tensorflow as tf
import sys

from utils import layers
from models.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        return logits

class HeteGAT_multi(tf.keras.Model):
    def __init__(self, nb_classes, nb_nodes, hid_units, n_heads, activation=tf.nn.elu, residual=False, mp_att_size=128):
        super(HeteGAT_multi, self).__init__()
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.mp_att_size = mp_att_size
        # Initialize layers here

    def call(self, inputs_list, bias_mat_list, training, attn_drop, ffd_drop):
        embed_list = []
        for inputs, bias_mat in zip(inputs_list, bias_mat_list):
            attns = []
            for _ in range(self.n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=self.hid_units[0], activation=self.activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=self.residual))
            h_1 = tf.concat(attns, axis=-1)
            embed_list.append(h_1)
        multi_embed = tf.concat(embed_list, axis=1)
        # Continue with the rest of the forward pass as defined previously
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, self.mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        logits_list = []
        for i in range(self.n_heads[-1]):
            head_output = tf.keras.layers.Dense(self.nb_nodes * self.nb_classes, activation=None)(final_embed)
            head_output = tf.reshape(head_output, [1, self.nb_nodes, self.nb_classes])
            logits_list.append(head_output)
        logits = tf.reduce_mean(tf.stack(logits_list), axis=0)
        logits = tf.reshape(logits, [-1, self.nb_nodes, self.nb_classes])
        return logits, final_embed, att_val

class HeteGAT_no_coef(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.keras.layers.Dense(nb_classes, activation=None)(final_embed))
        logits = tf.add_n(out) / n_heads[-1]
        print('de')
        logits = tf.expand_dims(logits, axis=0)
        return logits, final_embed, att_val

class HeteGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128,
                  return_coef=False):
        embed_list = []
        coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                if return_coef:
                    a1, a2 = layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef)
                    attns.append(a1)
                    head_coef_list.append(a2)
                else:
                    attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            head_coef = tf.concat(head_coef_list, axis=0)
            head_coef = tf.reduce_mean(head_coef, axis=0)
            coef_list.append(head_coef)
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.keras.layers.Dense(nb_classes, activation=None)(final_embed))
        logits = tf.add_n(out) / n_heads[-1]
        logits = tf.expand_dims(logits, axis=0)
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val
