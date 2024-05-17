from neo4j import GraphDatabase
import json

class ImportCVData:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_nodes_and_relationships(self, nodes, edges):
        with self.driver.session() as session:
            for node_id, node_data in nodes.items():
                session.execute_write(self._create_and_return_node, node_id, node_data)
            for edge in edges:
                session.execute_write(self._create_and_return_relationship, edge)

    @staticmethod
    def _create_and_return_node(tx, node_id, node_data):
        # Flatten the node_data dictionary into individual properties
        properties = ", ".join(f"{key}: ${key}" for key in node_data.keys())
        query = (
            f"CREATE (n:CVNode {{id: $node_id, {properties} }}) "
            "RETURN n"
        )
        # Pass the individual properties as parameters
        parameters = {key: value for key, value in node_data.items()}
        parameters["node_id"] = node_id
        result = tx.run(query, **parameters)
        return result.single()[0]

    @staticmethod
    def _create_and_return_relationship(tx, edge):
        query = (
            "MATCH (a:CVNode), (b:CVNode) "
            "WHERE a.id = $from_id AND b.id = $to_id "
            "CREATE (a)-[:CONNECTED_TO]->(b)"
        )
        tx.run(query, from_id=edge[0], to_id=edge[1])

if __name__ == "__main__":
    # Read structured data from file
    with open('structured_cv_data.txt', 'r') as file:
        data = file.read()
        data = json.loads(data.replace("'", "\""))

    nodes = data['Nodes']
    edges = data['Edges']

    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "devin_neo4j_password"  # Replace with the actual password

    importer = ImportCVData(uri, user, password)
    importer.create_nodes_and_relationships(nodes, edges)
    importer.close()
