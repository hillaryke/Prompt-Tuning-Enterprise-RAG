from app.utils.chat_models import ModelFactory
from langchain.schema import HumanMessage
from app.test_cases.models import TestCase

TEMPERATURE = 1.5
model_factory = ModelFactory(TEMPERATURE)

def generate_test_cases(task_description, amount=3):
    """Generates test cases using a language model based on the task description."""

    prompt_template = """
    You are a helpful AI assistant. Please generate {amount} diverse test cases for the following task:

    Task: {description}

    Each test case should include:
    * Scenario: A clear description of the situation or input to be tested.
    * Expected output: The ideal or expected output from the system.

    Format each test case like this:
    Scenario: [Scenario description]
    Expected output: [Expected output]
    """

    model = model_factory.get_chat_openai()

    messages = [
        HumanMessage(
            content=prompt_template.format(amount=amount, description=task_description)
        )
    ]

    response = model(messages)
    generated_text = response.content

    test_cases = []
    for case_str in generated_text.split("Scenario:"):
        case_str = case_str.strip() # Ensure that there are no spaces 
        if not case_str:
            continue
        try:
            scenario, expected_output = case_str.split("Expected output:", 1)
            test_cases.append(
                TestCase(scenario.strip(), expected_output.strip())
            )
        except ValueError:
            print(f"Skipping malformed test case: {case_str}")  # Handle cases where "Expected output:" is missing

    return test_cases