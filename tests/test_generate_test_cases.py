import unittest
from unittest.mock import patch, MagicMock
from app.test_cases.generate_test_cases import generate_test_cases
from app.test_cases.models import TestCase

class TestGenerateTestCases(unittest.TestCase):
  # @patch('app.utils.chat_models.ChatOpenAI')
  def test_generate_test_cases(self):
    # # Create a mock response
    # mock_response = MagicMock()
    # mock_response.content = """
    # Scenario: The function is given the numbers 1 and 2.
    # Expected output: The function returns 3.
    # Scenario: The function is given the numbers -1 and 2.
    # Expected output: The function returns 1.
    # Scenario: The function is given the numbers 0 and 0.
    # Expected output: The function returns 0.
    # """

    # # Configure the mock ChatOpenAI model to return the mock response
    # mock_chat_openai.return_value = mock_response

    task_description = "What is few shot learning."
    amount = 3

    test_cases = generate_test_cases(task_description, amount)

    # Check that the correct number of test cases were generated
    self.assertEqual(len(test_cases), amount)

    # Check that each test case is properly formatted
    for test_case in test_cases:
      self.assertIsInstance(test_case, TestCase)
      self.assertNotEqual(test_case.scenario, "")
      self.assertNotEqual(test_case.expected_output, "")

# Run the test
if __name__ == "__main__":
  unittest.main()