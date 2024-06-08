import nest_asyncio
from typing import List
from app.test_cases.generate_test_cases import TEMPERATURE
from app.utils.chat_models import ModelFactory
from app.test_cases.models import TestCase

from app.misc import Settings
from app.utils.docs_utils import format_docs_to_text
from app.utils.langchain_utils import generate_response_with_langchain

TEMPERATURE = 0.9
model_factory = ModelFactory()

human_message_prompt = """
    Test Cases: {formatted_test_cases_str}
    Task: {description}
    Context: {context_text}
    Number of prompts to generate: {num_prompt_candidates}.
"""

class PromptGenerator:
  def __init__(self, retriever, temperature: float = TEMPERATURE, model_name: str = None):
    self.model_name = model_name
    # self.system_message_prompt = system_message_prompt
    self.temperature = temperature
    self.retriever = retriever
    self.num_prompt_candidates = Settings.NUMBER_OF_PROMPT_CANDIDATES
    self.human_message_prompt = human_message_prompt
    self.llm_model = model_factory.get_chat_openai()

  def extract_prompts(self, response):
    lines = response.split("\n")
    prompts = [line.replace("Prompt: ", "").strip() for line in lines if line.startswith("Prompt: ")]
    return prompts

  def generate_prompt_candidates(self, testcases: List[TestCase], task_description: str) -> List[str]:
    """Generates prompt candidates using a chat language model."""

    # nest_asyncio.apply()

    formatted_test_cases_str = "\n".join(
      [
        f"Test case #{j+1}:\nScenario: {testcase.scenario}\nExpected output: {testcase.expected_output}"
        for j, testcase in enumerate(testcases)
      ]
    )

    context_docs = self.retriever.invoke(task_description)
    context_text = format_docs_to_text(context_docs)

    prompt_input_variables = {
        "description": task_description,
        "context_text": context_text,
        "num_prompt_candidates": self.num_prompt_candidates,
        "formatted_test_cases_str": formatted_test_cases_str
    }

    response = generate_response_with_langchain(
        system_message_prompt=Settings.PROMPT_CANDIDATES_GENERATION_SYSTEM_PROMPT,
        human_message_prompt=self.human_message_prompt,
        input_variables=prompt_input_variables,
        llm_model=self.llm_model
    )

    prompts = self.extract_prompts(response)

    return prompts