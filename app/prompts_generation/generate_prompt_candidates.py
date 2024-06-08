from multiprocessing import context
from os import system
from click import prompt
import nest_asyncio
from typing import List
from app.test_cases.generate_test_cases import TEMPERATURE
from app.utils.chat_models import ModelFactory
from app.test_cases.models import TestCase

from langchain.chat_models import ChatOpenAI  # Import the correct class for chat models
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.misc import Settings

TEMPERATURE = 0.9
model_factory = ModelFactory()

class PromptGenerator:
  def __init__(self, retriever, temperature: float = TEMPERATURE, model_name: str = None):
    self.model_name = model_name
    # self.system_message_prompt = system_message_prompt
    self.temperature = temperature
    self.retriever = retriever
    self.num_prompt_candidates = Settings.NUMBER_OF_PROMPT_CANDIDATES

  def extract_prompts(self, response):
    lines = response.split("\n")
    prompts = [line.replace("Prompt: ", "").strip() for line in lines if line.startswith("Prompt: ")]
    return prompts

  def generate_prompt_candidates(self, testcases: List[TestCase], task_description: str) -> List[str]:
    """Generates prompt candidates using a chat language model."""

    nest_asyncio.apply()

    formatted_test_cases_str = "\n".join(
      [
        f"Test case #{j+1}:\nScenario: {testcase.scenario}\nExpected output: {testcase.expected_output}"
        for j, testcase in enumerate(testcases)
      ]
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(Settings.PROMPT_CANDIDATE_GENERATION_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
        Test Cases: {formatted_test_cases_str}
        Task: {description}
        Context: {context_text}
        Number of prompts to generate: {num_prompt_candidates}.
        """
    )

    llm = model_factory.get_chat_openai()

    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = chat_prompt_template | llm | StrOutputParser()

    context_docs = self.retriever.invoke(task_description)

        # Post-processing
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    context_text = format_docs(context_docs)

    prompt_input_variables = {
        "description": task_description,
        "context_text": context_text,
        "num_prompt_candidates": self.num_prompt_candidates,
        "formatted_test_cases_str": formatted_test_cases_str
    }

    response = chain.invoke(prompt_input_variables)

    prompts = self.extract_prompts(response)


    return prompts