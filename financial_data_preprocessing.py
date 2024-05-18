import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle  # Import pickle to save scaler object

def load_data(file_path):
    """
    Load the historical Bitcoin price data from a CSV file.

    Parameters:
    - file_path: str, the path to the CSV file.

    Returns:
    - df: DataFrame, the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def select_features(df):
    """
    Select and return relevant features for the prediction task.

    Parameters:
    - df: DataFrame, the loaded data.

    Returns:
    - df_selected: DataFrame, the selected features.
    """
    # For simplicity, we'll use 'Close' and 'Volume' as features
    df_selected = df[['Close', 'Volume']]
    return df_selected

def normalize_features(df, scaler):
    """
    Normalize the features using MinMaxScaler.

    Parameters:
    - df: DataFrame, the selected features.
    - scaler: MinMaxScaler, the scaler used for normalization.

    Returns:
    - df_normalized: DataFrame, the normalized features.
    """
    df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_normalized

def construct_feature_matrix(df):
    """
    Construct the feature matrix from the DataFrame and pad it to have a number of rows that is a multiple of 16.

    Parameters:
    - df: DataFrame, the normalized features.

    Returns:
    - feature_matrix: ndarray, the feature matrix.
    """
    feature_matrix = df.values
    # Calculate the number of padding rows needed to reach the next multiple of 16
    padding_rows = (-len(df)) % 16
    if padding_rows > 0:
        # Pad the feature matrix with zeros
        padding = np.zeros((padding_rows, feature_matrix.shape[1]))
        feature_matrix = np.vstack([feature_matrix, padding])
    return feature_matrix

def construct_adjacency_matrix(df):
    """
    Construct the adjacency matrix for the graph neural network.

    For the Bitcoin price prediction task, we will use a simple identity matrix,
    treating each time step as disconnected.

    Parameters:
    - df: DataFrame, the normalized features.

    Returns:
    - adjacency_matrix: ndarray, the adjacency matrix.
    """
    adjacency_matrix = np.identity(len(df))
    return adjacency_matrix

def save_data(file_name, data):
    """
    Save data to a pickle file.

    Parameters:
    - file_name: str, the name of the file to save the data to.
    - data: object, the data to save.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def generate_labels(df, prediction_days=1):
    """
    Generate labels for the training data.

    Parameters:
    - df: DataFrame, the loaded data.
    - prediction_days: int, the number of days ahead to predict.

    Returns:
    - labels: ndarray, the labels for the training data.
    """
    # Shift the 'Close' price column to create the target variable
    # representing the future price we want to predict
    labels = df['Close'].shift(-prediction_days).values
    # Fill 'nan' values using forward fill method
    labels = pd.Series(labels).fillna(method='ffill').values
    return labels

def prepare_input_data_for_prediction(input_data_path, scaler_path):
    """
    Prepare the input data for prediction by normalizing and constructing the feature matrix.

    Parameters:
    - input_data_path: str, the path to the input data CSV file.
    - scaler_path: str, the path to the saved scaler pickle file.

    Returns:
    - feature_matrix: ndarray, the feature matrix for prediction.
    - adjacency_matrix: ndarray, the adjacency matrix for prediction.
    """
    # Load the input data
    input_data = load_data(input_data_path)
    if input_data is None:
        return None, None

    # Load the scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"The scaler file {scaler_path} was not found.")
        return None, None

    # Select features
    input_data_selected = select_features(input_data)

    # Normalize features
    input_data_normalized = normalize_features(input_data_selected, scaler)

    # Construct feature matrix
    feature_matrix = construct_feature_matrix(input_data_normalized)

    # Construct adjacency matrix
    adjacency_matrix = construct_adjacency_matrix(input_data_normalized)

    return feature_matrix, adjacency_matrix

if __name__ == "__main__":
    # Update the file path with the actual path to the downloaded Bitcoin historical data CSV file
    file_path = '/home/ubuntu/browser_downloads/BTC-USD.csv'
    scaler_path = 'scaler.pkl'  # Path to the scaler used during training

    # Prepare input data for prediction
    feature_matrix, adjacency_matrix = prepare_input_data_for_prediction(file_path, scaler_path)

    # Save the feature matrix and adjacency matrix for prediction
    save_data('input_data_2024_feature_matrix.pkl', feature_matrix)
    save_data('input_data_2024_adjacency_matrix.pkl', adjacency_matrix)
