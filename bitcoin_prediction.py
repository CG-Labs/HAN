import yfinance as yf
from datetime import datetime
import tensorflow as tf
from models.gat import HeteGAT
from utils.process import process

# Function to fetch historical Bitcoin data
def fetch_bitcoin_data(start_date, end_date):
    try:
        # Fetch historical data for Bitcoin from Yahoo Finance
        data = yf.download('BTC-USD', start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    try:
        # TODO: Implement actual preprocessing steps based on the model's requirements
        # For now, we'll just return the data as is
        return data
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        return None

# Function to make predictions using the model
def make_prediction(model, data):
    try:
        # TODO: Update this function to format input data correctly for the model
        # This function will use the trained model to make predictions
        return model.predict(data)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# Function to load the trained model
def load_model(model_path):
    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(model_path, custom_objects={'HeteGAT': HeteGAT})
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

# Main execution
if __name__ == "__main__":
    try:
        # Define the dates for the predictions
        end_of_may_2024 = datetime(2024, 5, 31)
        end_of_december_2024 = datetime(2024, 12, 31)

        # Fetch and preprocess the data
        may_data = preprocess_data(fetch_bitcoin_data('2020-01-01', end_of_may_2024.strftime('%Y-%m-%d')))
        december_data = preprocess_data(fetch_bitcoin_data('2020-01-01', end_of_december_2024.strftime('%Y-%m-%d')))

        if may_data is not None and december_data is not None:
            # TODO: Replace the placeholder path with the actual path to the trained model
            model_path = 'path_to_trained_model.h5'  # Replace with the actual path to the trained model

            # Load the trained model
            model = load_model(model_path)

            if model is not None:
                # Make predictions
                may_prediction = make_prediction(model, may_data)
                december_prediction = make_prediction(model, december_data)

                # Output the predictions
                print(f"Bitcoin price prediction for end of May 2024: {may_prediction}")
                print(f"Bitcoin price prediction for end of December 2024: {december_prediction}")
            else:
                print("Failed to load the model.")
        else:
            print("Failed to fetch or preprocess the data.")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")

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
