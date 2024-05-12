import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_feature_vector(section_content):
    """
    Create a feature vector from the section content using TF-IDF.
    """
    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Tokenize and build vocab
    vectorizer.fit(section_content)

    # Encode document
    vector = vectorizer.transform(section_content)

    # Summarize encoded vector
    return vector.toarray()

def create_adjacency_matrix(cv_data):
    """
    Create an adjacency matrix from the CV data using cosine similarity.
    """
    # Extract all section contents and vectorize them
    contents = list(cv_data.values())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(contents)

    # Compute cosine similarity between vectors
    cosine_matrix = cosine_similarity(vectors)

    # Convert cosine similarity matrix to adjacency matrix format
    adjacency_matrix = {}
    sections = list(cv_data.keys())
    for i, section in enumerate(sections):
        adjacency_matrix[section] = {}
        for j, related_section in enumerate(sections):
            # We consider a threshold to determine if there is a link or not
            adjacency_matrix[section][related_section] = 1 if cosine_matrix[i][j] > 0.5 else 0

    return adjacency_matrix

def process_cv_data(cv_data):
    """
    Process the CV data to create feature vectors and an adjacency matrix.
    """
    feature_vectors = {}
    for section, content in cv_data.items():
        feature_vectors[section] = create_feature_vector([content])

    adjacency_matrix = create_adjacency_matrix(cv_data)

    return feature_vectors, adjacency_matrix

if __name__ == "__main__":
    # Example CV data structure
    cv_data = {
        'PERSONAL_PROFILE': 'Example personal profile content\nAnother line of content',
        'KEY_SKILLS': 'Example key skills content\nAnother line of content',
        'PROFESSIONAL_EXPERIENCE': 'Example professional experience content\nAnother line of content',
        'EDUCATION': 'Example education content\nAnother line of content'
    }

    # Process the CV data
    feature_vectors, adjacency_matrix = process_cv_data(cv_data)

    # For demonstration, print the feature vectors and adjacency matrix
    print("Feature Vectors:")
    for section, features in feature_vectors.items():
        print(f"{section}: {features}")
    print("\nAdjacency Matrix:")
    for section, connections in adjacency_matrix.items():
        print(f"{section}: {connections}")
