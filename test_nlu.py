import unittest
from nlu import NaturalLanguageUnderstanding
from state_machine import StateMachine

class TestNaturalLanguageUnderstanding(unittest.TestCase):
    def setUp(self):
        self.nlu = NaturalLanguageUnderstanding()
        self.sm = StateMachine(num_classes=3, num_features=10)  # Example values

    def test_parse_input(self):
        self.assertEqual(self.nlu.parse_input(" Analyze CV "), "analyze cv")
        self.assertEqual(self.nlu.parse_input(" PREDICT Market Trends "), "predict market trends")

    def test_extract_intent(self):
        self.assertEqual(self.nlu.extract_intent("analyze cv"), 'analyze_cv')
        self.assertEqual(self.nlu.extract_intent("predict market trends"), 'predict_market_trends')
        self.assertEqual(self.nlu.extract_intent("complete analysis"), 'complete_analysis')
        self.assertEqual(self.nlu.extract_intent("unknown command"), 'unknown')

    def test_get_action(self):
        self.assertEqual(self.nlu.get_action('analyze_cv'), 'retrieve_data')
        self.assertEqual(self.nlu.get_action('predict_market_trends'), 'make_prediction')
        self.assertEqual(self.nlu.get_action('complete_analysis'), 'complete_analysis')
        self.assertIsNone(self.nlu.get_action('unknown'))

    def test_state_machine_integration(self):
        # Test state machine transitions based on NLU output
        self.sm.process_input("Analyze CV")
        self.assertEqual(self.sm.current_state, 'data_retrieval')
        self.sm.process_input("Predict Market Trends")
        self.assertEqual(self.sm.current_state, 'prediction')
        self.sm.process_input("Complete Analysis")
        self.assertEqual(self.sm.current_state, 'completed')
        # Test reset functionality
        self.sm.reset()
        self.assertEqual(self.sm.current_state, 'initial')

if __name__ == '__main__':
    unittest.main()
