import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging configuration to capture debug statements
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def extract_sections(cv_text):
    """
    Extracts sections from the CV text using regular expressions to identify headings.
    """
    sections = {}
    current_section = None
    content = []
    for line in cv_text.split('\n'):
        # Check for section headings
        if re.match(r'^[A-Z ]+$', line.strip()):
            # Save previous section and content if exists
            if current_section:
                sections[current_section] = '\n'.join(content).strip()
                content = []
            current_section = line.strip()
        else:
            content.append(line)
    # Save the last section
    if current_section:
        sections[current_section] = '\n'.join(content).strip()
    return sections

def create_feature_vector(section_content, vectorizer):
    """
    Create a feature vector from the section content using a pre-fitted TF-IDF vectorizer.
    """
    # Encode document using the pre-fitted vectorizer
    vector = vectorizer.transform([section_content])

    # Return the encoded vector as a 2D array with one row
    return vector.toarray()

def create_adjacency_matrix(cv_data):
    """
    Create an adjacency matrix from the CV data using cosine similarity.
    """
    # Extract all section contents and vectorize them
    contents = list(cv_data.values())
    # Initialize a TF-IDF Vectorizer with lowered min_df and without stop words
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
    vectors = vectorizer.fit_transform(contents)

    # Compute cosine similarity between vectors
    cosine_matrix = cosine_similarity(vectors)

    # Convert cosine similarity matrix to binary adjacency matrix format
    adjacency_matrix = (cosine_matrix > 0.5).astype(int)

    # Return the adjacency matrix wrapped in a list
    return [adjacency_matrix]

def process_cv_data(cv_text):
    """
    Process the CV text to create feature vectors and an adjacency matrix.
    """
    # Extract sections from CV text
    cv_data = extract_sections(cv_text)

    # Collect all section contents
    all_contents = [content for content in cv_data.values() if content.strip()]

    # Initialize a TF-IDF Vectorizer with lowered min_df and without stop words
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None)

    # Fit the vectorizer on the entire corpus of section contents
    vectorizer.fit(all_contents)

    feature_vectors = []
    for section, content in cv_data.items():
        # Ensure that the content is not empty before creating feature vector
        if content.strip():
            # Create feature vector using the pre-fitted vectorizer
            feature_vectors.append(create_feature_vector(content, vectorizer))
            logging.debug("Appended feature vector type: %s, shape: %s", type(feature_vectors[-1]), feature_vectors[-1].shape)
        # If the content is empty, skip adding it to the feature_vectors list

    adjacency_matrix = create_adjacency_matrix(cv_data)

    # Generate dummy y_train, y_val, y_test, train_mask, val_mask, test_mask
    # Assuming 3 classes for the purpose of generating dummy data
    nb_classes = 3
    nb_nodes = len(feature_vectors)
    y_train = np.zeros((nb_nodes, nb_classes))
    y_val = np.zeros((nb_nodes, nb_classes))
    y_test = np.zeros((nb_nodes, nb_classes))
    train_mask = np.zeros((nb_nodes,)).astype(bool)
    val_mask = np.zeros((nb_nodes,)).astype(bool)
    test_mask = np.zeros((nb_nodes,)).astype(bool)

    # For simplicity, let's say the first node is for training, the second for validation, and the third for testing
    if nb_nodes > 2:
        y_train[0, 0] = 1  # Class 0
        y_val[1, 1] = 1    # Class 1
        y_test[2, 2] = 1   # Class 2
        train_mask[0] = True
        val_mask[1] = True
        test_mask[2] = True

    # Log the length of feature_vectors for debugging
    print("Length of feature_vectors:", len(feature_vectors))
    logging.debug("Type of feature_vectors before return: %s", type(feature_vectors))
    return feature_vectors, adjacency_matrix, y_train, y_val, y_test, train_mask, val_mask, test_mask

if __name__ == "__main__":
    # Read the CV text file
    with open('Alan_Woulfe_CV.txt', 'r') as file:
        cv_text = file.read()

    # Process the CV text
    feature_vectors, adjacency_matrix, y_train, y_val, y_test, train_mask, val_mask, test_mask = process_cv_data(cv_text)

    # For demonstration, print the feature vectors and adjacency matrix
    print("Feature Vectors:")
    for features in feature_vectors:
        print(features)
    print("\nAdjacency Matrix:")
    for section, connections in adjacency_matrix.items():
        print(f"{section}: {connections}")
