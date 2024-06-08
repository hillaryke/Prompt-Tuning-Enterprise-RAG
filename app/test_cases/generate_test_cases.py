from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from app.misc.settings import Settings
from app.utils.chat_models import ModelFactory
from app.test_cases.models import TestCase
from app.utils.docs_utils import format_docs_to_text
from app.utils.langchain_utils import generate_response_with_langchain


TEMPERATURE = 1.5
model_factory = ModelFactory(TEMPERATURE)

human_message_prompt = """
    Please generate {amount} diverse test cases for the following task:
    Task (task description): {task_description} 
    Context: {context}
"""

def generate_test_cases(task_description, retriever, amount=3):
    """Generates test cases using a language model based on the task description."""

    llm_model = model_factory.get_chat_openai()

    context_docs = retriever.invoke(task_description)
    context_text = format_docs_to_text(context_docs)

    prompt_input_variables = {
        "task_description": task_description, 
        "context": context_text,
        "amount": amount
    }

    generated_text = generate_response_with_langchain(
        system_message_prompt=Settings.CREATE_TEST_CASES_SYSTEM_PROMPT,
        human_message_prompt=human_message_prompt,
        input_variables=prompt_input_variables,
        llm_model=llm_model
    )

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