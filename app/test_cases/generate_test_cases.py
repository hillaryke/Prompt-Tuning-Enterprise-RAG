from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from app.utils.chat_models import ModelFactory
from langchain.schema import HumanMessage
from app.test_cases.models import TestCase

TEMPERATURE = 1.5
model_factory = ModelFactory(TEMPERATURE)

create_test_case_prompt = """
        Your job is to create a test case for a given task and it's expected output. The task is a description of a use-case.

        It should be general enough that it can be used to test the AI's ability to perform the task in retreiving documents.
        It should never actually complete the task, but it should be a good example of the task.

        Example:
        Task: Creates a landing page headline for a new product
        Test case: "A new type of toothpaste that whitens teeth in 5 minutes"
        Test case: "A fitness app that helps you lose weight"
        Test case: "A therapist for dogs"

        Task: Generates a title for a blog post that will get the most clicks
        Test case: "How to increase your productivity by 10x"
        Test case: "A post about the best travel destinations in the world"
        Test case: "The best restaurants in New York"

        Task: Generate a paragraph that describes a product
        Test case: "The new Macbook Pro"
        Test case: "Nike shoes"
        Test case: "A case for iPhones that's velvety smooth and very durable"

        You will be graded based on the performance of your test case and expected output... but don't cheat! You cannot include specifics about the task in your test case. Any test cases with examples will be disqualified.
        Be really creative! The most creative test cases will be rewarded.

        YOU NEVER OUTPUT SOMETHING THAT COMPLETES THE TASK. ONLY A TEST CASE AND ITS EXPECTED OUTPUT.

        Most importantly, output NOTHING but the test case and expected output. Do not include anything else in your message.
        Each test case should include:
        * Scenario: A clear description of the situation or input to be tested.
        * Expected output: The ideal or expected output from the system.
"""

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

    llm = model_factory.get_chat_openai()

    system_message_prompt = SystemMessagePromptTemplate.from_template(create_test_case_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
            Please generate {amount} diverse test cases for the following task:
            Task (task description): {task_description} 
            Then the program should output:

            Remember, always output only the test cases and exact expected output for each test case.

            Format each test case like this:
            Scenario: [Scenario description]
            Expected output: [Expected output]
        """
    )

    # messages = [
    #     HumanMessage(
    #         content=prompt_template.format(amount=amount, description=task_description)
    #     )
    # ]

    # response = model.invoke(messages)
    # generated_text = response.content
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = chat_prompt_template | llm | StrOutputParser()

    generated_text = chain.invoke({"task_description": task_description, "amount": amount})

    print(generated_text)

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