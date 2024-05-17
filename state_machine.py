# state_machine.py
# This script serves as the basis for the agentic system's state machine.
# It will handle various states of user intent and orchestrate the necessary tasks.

from neo4j_integration import Neo4jConnection
from process_cv_data import process_cv_data
from gnn_model import GNNModel
from nlu import NaturalLanguageUnderstanding

# Neo4j connection URL and credentials
neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "Devin2023!"

class StateMachine:
    """
    The StateMachine class manages the different states of the application and transitions between them based on user actions.

    It integrates with the Neo4j database for data retrieval, the GNNModel for data analysis and prediction, and the NaturalLanguageUnderstanding class for interpreting user input.

    Attributes:
        states (list): A list of the possible states the machine can be in.
        transitions (dict): A dictionary where each key is a state and the value is another dictionary mapping actions to resulting states.
        current_state (str): The current state of the machine.
        neo4j_integration (Neo4jConnection): An instance of the Neo4jConnection class for interacting with the Neo4j database.
        gnn_model (GNNModel): An instance of the GNNModel class for analyzing data and making predictions.
        data (any): The data retrieved from the Neo4j database.
        analysis_results (any): The results of analyzing the data with the GNNModel.
    """
    def __init__(self, num_classes, num_features):
        self.states = ['initial', 'data_retrieval', 'data_analysis', 'prediction', 'idle', 'processing', 'executing', 'completed']
        self.transitions = {
            'initial': {
                'retrieve_data': 'data_retrieval',
                'analyze_data': 'data_analysis',
                'make_prediction': 'prediction'
            },
            'data_retrieval': {
                'analyze_data': 'data_analysis',
                'make_prediction': 'prediction'
            },
            'data_analysis': {
                'make_prediction': 'prediction'
            },
            'prediction': {
                'complete_analysis': 'completed',  # Transition to 'completed' after analysis is complete
                'complete': 'initial'
            },
            'idle': {
                'process_input': 'processing'
            },
            'processing': {
                'execute_task': 'executing'
            },
            'executing': {
                'finish_task': 'completed'
            },
            'completed': {
                'reset': 'idle'
            }
        }
        self.current_state = 'initial'
        self.neo4j_integration = Neo4jConnection(neo4j_url, neo4j_username, neo4j_password)
        self.gnn_model = GNNModel(num_classes=num_classes, num_features=num_features)
        self.data = None
        self.analysis_results = None

    def transition(self, action):
        """
        Transition the state machine to a new state based on the given action.

        Parameters:
        - action (str): The action to be executed, which causes a state transition.

        If the action is valid for the current state, the state is updated. Otherwise,
        an error message is printed indicating the invalid action.
        """
        if action in self.transitions[self.current_state]:
            self.current_state = self.transitions[self.current_state][action]
            print(f"Transitioned to {self.current_state}")
        else:
            print(f"Action {action} is not valid from state {self.current_state}")

    def execute_action(self, action, data=None, analysis_results=None):
        """
        Execute actions based on the current state and user input.

        Parameters:
        - action (str): The action to be executed.
        - data (any, optional): The data to be used for analysis.
        - analysis_results (any, optional): The results of data analysis to be used for prediction.

        Depending on the current state, this method may connect to the Neo4j database to retrieve data,
        use the GNN model to analyze data, or make predictions based on analyzed data.
        If the action is not valid for the current state, an error message is printed.
        """
        # Execution logic based on the current state...
        if self.current_state == 'data_retrieval':
            # Connect to Neo4j database using credentials
            # Retrieve relevant data for analysis
            try:
                self.data = self.neo4j_integration.retrieve_data()
                print("Data retrieval successful.")
            except Exception as e:
                print(f"Error retrieving data: {e}")
        elif self.current_state == 'data_analysis':
            # Logic to analyze the retrieved data
            try:
                if data is None:
                    raise ValueError("No data provided for analysis.")
                self.analysis_results = self.gnn_model.analyze_data(data)
                print("Data analysis successful.")
            except Exception as e:
                print(f"Error analyzing data: {e}")
        elif self.current_state == 'prediction':
            # Logic to make predictions based on analyzed data
            try:
                if analysis_results is None:
                    raise ValueError("No analysis results provided for prediction.")
                predictions = self.gnn_model.make_prediction(analysis_results)
                print("Predictions made successfully.")
                return predictions
            except Exception as e:
                print(f"Error making predictions: {e}")
        else:
            print(f"No execution logic defined for action {action} in state {self.current_state}")

    def process_input(self, user_input):
        """
        Process user input to determine the appropriate action and initiate state transitions.

        Parameters:
        - user_input (str): The raw input from the user.

        This method uses the NaturalLanguageUnderstanding class to parse the input,
        extract the intent, and determine the corresponding action. If the action is valid
        for the current state, a state transition is initiated.
        """
        # Processing logic using NaturalLanguageUnderstanding...
        nlu = NaturalLanguageUnderstanding()
        processed_input = nlu.parse_input(user_input)
        intent = nlu.extract_intent(processed_input)
        action = nlu.get_action(intent)

        if action and action in self.transitions[self.current_state]:
            self.transition(action)
        else:
            print(f"Unrecognized or invalid action for the current state: {self.current_state}")

    def reset(self):
        """
        Reset the state machine to the initial state and clear any stored data and analysis results.

        This method reinitializes the state machine, setting the current state to 'initial'
        and clearing the 'data' and 'analysis_results' attributes.
        """
        # Reset logic...
        self.current_state = 'initial'
        self.data = None
        self.analysis_results = None
        print("State machine has been reset to the initial state.")

# Example usage
if __name__ == "__main__":
    num_classes = 2  # Example value, should be set based on the actual task
    num_features = 34  # Example value, should be set based on the actual data
    sm = StateMachine(num_classes=num_classes, num_features=num_features)
    # Simulate user input
    user_input = "analyze CV"
    sm.process_input(user_input)
    sm.execute_action('retrieve_data')
    sm.transition('analyze_data')
    sm.execute_action('analyze_data', data=sm.data)
    sm.transition('make_prediction')
    predictions = sm.execute_action('make_prediction', analysis_results=sm.analysis_results)
    print(predictions)
    # Reset the state machine
    sm.reset()
