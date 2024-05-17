import logging
from neo4j_integration import Neo4jConnection

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_retrieve_data():
    # Create a Neo4j connection instance
    neo4j_conn = Neo4jConnection('bolt://localhost:7687', 'neo4j', 'Devin2023!')

    # Retrieve data from Neo4j
    data = neo4j_conn.retrieve_data()

    # Print the retrieved data
    print(data)

if __name__ == "__main__":
    test_retrieve_data()
