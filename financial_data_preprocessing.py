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

def normalize_features(df):
    """
    Normalize the features using MinMaxScaler.

    Parameters:
    - df: DataFrame, the selected features.

    Returns:
    - df_normalized: DataFrame, the normalized features.
    - scaler: MinMaxScaler, the scaler used for normalization.
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized, scaler

def construct_feature_matrix(df):
    """
    Construct the feature matrix from the DataFrame.

    Parameters:
    - df: DataFrame, the normalized features.

    Returns:
    - feature_matrix: ndarray, the feature matrix.
    """
    feature_matrix = df.values
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

if __name__ == "__main__":
    # Update the file path with the actual path to the downloaded Bitcoin historical data CSV file
    file_path = '~/browser_downloads/BTC-USD.csv'

    # Load the data
    df = load_data(file_path)
    if df is not None:
        # Select features
        df_selected = select_features(df)

        # Normalize features
        df_normalized, scaler = normalize_features(df_selected)

        # Construct feature matrix
        feature_matrix = construct_feature_matrix(df_normalized)

        # Construct adjacency matrix
        adjacency_matrix = construct_adjacency_matrix(df_normalized)

        # Save the scaler for later use
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # These matrices can now be used to train the GNN model
