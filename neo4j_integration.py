from neo4j import GraphDatabase
import logging
from process_cv_data import process_cv_data
from docx import Document

# Neo4j connection URL and credentials
neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "neo4j"

class Neo4jConnection:
    """
    Neo4j Database Connection
    """
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            logging.error("Failed to create a database connection: %s", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            logging.error("Query failed: %s", e)
        finally:
            if session is not None:
                session.close()
        return response

def create_nodes_and_relationships(feature_vectors, adjacency_matrix):
    # Connect to Neo4j
    conn = Neo4jConnection(neo4j_url, neo4j_username, neo4j_password)

    # Create nodes
    for i, features in enumerate(feature_vectors):
        conn.query("CREATE (n:CVSection {id: $id, features: $features})", parameters={'id': i, 'features': features.tolist()})

    # Create relationships
    for i, connections in enumerate(adjacency_matrix):
        for j, connected in enumerate(connections):
            if connected == 1:
                conn.query("MATCH (a:CVSection {id: $id1}), (b:CVSection {id: $id2}) CREATE (a)-[:CONNECTED_TO]->(b)", parameters={'id1': i, 'id2': j})

    # Close the connection
    conn.close()

if __name__ == "__main__":
    # Read the CV from a .docx file
    with open('Alan_Woulfe_CV.docx', 'rb') as file:
        document = Document(file)
        cv_text = "\n".join([paragraph.text for paragraph in document.paragraphs])

    # Process the CV text
    feature_vectors, adjacency_matrix, _, _, _, _, _, _ = process_cv_data(cv_text)

    # Create nodes and relationships in Neo4j
    create_nodes_and_relationships(feature_vectors, adjacency_matrix)
