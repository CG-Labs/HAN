import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class ConcatLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)

class AddLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.add_n(inputs)

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False, return_shapes=False):
        attns = []
        shapes_list = []
        for _ in range(n_heads[0]):
            attn, shapes = layers.attn_head(inputs, bias_mat=bias_mat,
                                            out_sz=hid_units[0], activation=activation,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                            return_shapes=True)  # Set return_shapes to True to get shapes for debugging
            attns.append(attn)
            shapes_list.append(shapes)
        h_1 = ConcatLayer(axis=-1)(attns)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attn, shapes = layers.attn_head(h_1, bias_mat=bias_mat,
                                                out_sz=hid_units[i], activation=activation,
                                                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual,
                                                return_shapes=True)  # Set return_shapes to True to get shapes for debugging
                attns.append(attn)
                shapes_list.append(shapes)
            h_1 = ConcatLayer(axis=-1)(attns)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                        return_shapes=True)[0])  # Set return_shapes to True to get shapes for debugging
        logits = AddLayer()([out]) / n_heads[-1]

        if return_shapes:
            return logits, shapes_list
        else:
            return logits

class HeteGAT_multi(BaseGAttN):
    def inference(inputs_list, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        print('Debug: Starting attention head computations for first layer...')
        for inputs, bias_mat in zip(inputs_list, bias_mat_list):
            attns = []
            jhy_embeds = []
            for _ in range(n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
            h_1 = ConcatLayer(axis=-1)(attns)
            print('Debug: Attention head computations for first layer completed.')

        print('Debug: Starting attention head computations for subsequent layers...')
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i],
                                              activation=activation,
                                              in_drop=ffd_drop,
                                              coef_drop=attn_drop, residual=residual))
            h_1 = ConcatLayer(axis=-1)(attns)
        print('Debug: Attention head computations for subsequent layers completed.')

        print('Debug: Concatenating embeddings from different types...')
        multi_embed = ConcatLayer(axis=1)(embed_list)
        print('Debug: Concatenation completed.')

        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)

        print('Debug: Computing final logits...')
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.keras.layers.Dense(nb_classes, activation=None)(final_embed))
        logits = AddLayer()([out]) / n_heads[-1]
        print('Debug: Final logits computation completed.')

        logits = tf.expand_dims(logits, axis=0)
        return logits, final_embed, att_val

class HeteGAT_no_coef(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        # coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):

                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            h_1 = ConcatLayer(axis=-1)(attns)
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
                h_1 = ConcatLayer(axis=-1)(attns)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = ConcatLayer(axis=1)(embed_list)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = AddLayer()([out]) / n_heads[-1]
        # logits_list.append(logits)
        print('de')
        logits = tf.expand_dims(logits, axis=0)
        # if return_coef:
        #     return logits, final_embed, att_val, coef_list
        # else:
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
                    # attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                               out_sz=hid_units[0], activation=activation,
                    #                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                    #                               return_coef=return_coef)[0])
                    #
                    # head_coef_list.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                                        out_sz=hid_units[0], activation=activation,
                    #                                        in_drop=ffd_drop, coef_drop=attn_drop,
                    #                                        residual=False,
                    #                                        return_coef=return_coef)[1])
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
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = AddLayer()([out]) / n_heads[-1]
        # logits_list.append(logits)
        logits = tf.expand_dims(logits, axis=0)
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val
