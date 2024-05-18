import docx
import re
from collections import defaultdict

def read_cv(file_path):
    """
    Reads a CV from a .docx file and extracts text.
    :param file_path: str, path to the .docx CV file
    :return: str, extracted text from the CV
    """
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"An error occurred while reading the CV: {e}")
        return None

def extract_features(cv_text):
    """
    Extracts features from CV text using regular expressions.
    :param cv_text: str, the full text of the CV
    :return: dict, a dictionary of extracted features
    """
    features = defaultdict(list)
    # Updated regex patterns for extracting features
    patterns = {
        'name': r'Name:\s*(.*)',
        'experience': r'Experience:\s*(.*)',
        'education': r'Education:\s*(.*)',
        'skills': r'Skills:\s*(.*)',
        'certifications': r'Certifications:\s*(.*)',
        'publications': r'Publications:\s*(.*)'
    }

    for feature, pattern in patterns.items():
        matches = re.findall(pattern, cv_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            features[feature].append(match.strip())

    return dict(features)

def construct_feature_matrix(features):
    """
    Constructs a feature matrix from extracted features.
    :param features: dict, a dictionary of extracted features
    :return: list, a list of lists representing the feature matrix
    """
    # Updated feature matrix construction
    feature_matrix = []
    for key, items in features.items():
        if key == 'skills':
            # Assuming skills are separated by commas
            skills_vector = [0] * 100  # Example fixed size of skills vector
            for skill in items:
                skill_list = skill.split(',')
                for skill in skill_list:
                    index = hash(skill) % 100  # Simple hash function example
                    skills_vector[index] = 1
            feature_matrix.append(skills_vector)
        else:
            # Other features can be represented as the count of occurrences
            feature_matrix.append([len(items)])
    return feature_matrix

def construct_adjacency_matrix(features):
    """
    Constructs an adjacency matrix from extracted features.
    :param features: dict, a dictionary of extracted features
    :return: list, a list of lists representing the adjacency matrix
    """
    # Updated adjacency matrix construction
    adjacency_matrix = [[0] * len(features) for _ in range(len(features))]
    for i, feature_items_i in enumerate(features.values()):
        for j, feature_items_j in enumerate(features.values()):
            if i != j and set(feature_items_i).intersection(set(feature_items_j)):
                adjacency_matrix[i][j] = 1  # Connection between different features
            else:
                adjacency_matrix[i][j] = 0
    return adjacency_matrix

# Example usage
if __name__ == "__main__":
    cv_path = "path_to_cv.docx"  # Replace with actual path to a CV document
    cv_text = read_cv(cv_path)
    if cv_text:
        features = extract_features(cv_text)
        feature_matrix = construct_feature_matrix(features)
        adjacency_matrix = construct_adjacency_matrix(features)
        print("Feature Matrix:", feature_matrix)
        print("Adjacency Matrix:", adjacency_matrix)
