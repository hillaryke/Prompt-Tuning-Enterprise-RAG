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

TEMPERATURE = 0.9
model_factory = ModelFactory()

class PromptGenerator:
  def __init__(self, temperature: float = TEMPERATURE, model_name: str = None):
    self.model_name = model_name
    # self.system_message_prompt = system_message_prompt
    self.temperature = temperature
    self.num_prompt_candidates = 3

  def extract_prompts(self, response):
    lines = response.split("\n")
    prompts = [line.replace("Prompt: ", "").strip() for line in lines if line.startswith("Prompt: ")]
    return prompts

  def generate_prompt_candidates(self, testcases: List[TestCase], task_description: str) -> List[str]:
    """Generates prompt candidates using a chat language model."""

    nest_asyncio.apply()

    candidates = []

    # for i in range(5):
      # system_message = self.system_messages[i % len(self.system_messages)]

    formatted_test_cases_str = "\n".join(
      [
        f"Test case #{j+1}:\nScenario: {testcase.scenario}\nExpected output: {testcase.expected_output}"
        for j, testcase in enumerate(testcases)
      ]
    )

    system_message_prompt = """  
        You are an AI-powered natural language processing expert in information retrieval. Your role is to provide advanced techniques and algorithms for generating superior prompts that optimize user queries and ensure the best performance of automatic prompt generation. Your expertise lies in understanding user intent, analyzing query patterns, and generating contextually relevant prompts that enable efficient and accurate retrieval of information. With your skills and abilities, you are capable of fine-tuning models to enhance prompt generation, leveraging semantic understanding and query understanding to deliver optimal results. By utilizing cutting-edge techniques in the field, you can generate automatic prompts that empower users to obtain the most relevant and comprehensive information for their queries.

        Your task is to formulate exactly the asked number of prompts prompts from the provided original prompt that are better and using the given context.
        
        The generated prompt must satisfy the rules given below:
        0. The generated prompted should only contain the prompt and no numbering
        1.The prompt should make sense to humans even when read without the given context.
        2.The prompt should be fully created from the given context.
        3.The prompt should be framed from a part of context that contains important information. It can also be from tables,code,etc.
        4.The prompt must be reasonable and must be understood and responded by humans.
        5.Do no use phrases like 'provided context',etc in the prompt
        6.The prompt should not contain more than 15 words, make of use of abbreviation wherever possible.
        7.The prompt should not be a verbatim copy of the context.
        8.The prompt should not include double quotes, instead just given it as is.
    """


    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
        You are to use these test cases to generate the prompts:
        {formatted_test_cases_str}

        Here is what the user wants the final prompt to accomplish:
        Task: {description}
        
        Number of prompts to generate: {num_prompt_candidates}.

        ENSURE YOU FOLLOW THIS FORMAT EXACTLY WHEN RETURNING PROMPTS. START WITH 'Prompt:' AND THEN THE PROMPT. 
        DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE.:
        Prompt: <Generated Prompt 1>
        Prompt: <Generated Prompt 2>
        Prompt: <Generated Prompt 3>
        ...

        Respond with the prompts, and nothing else. Be creative.
        ENSURE THE PROMPT SOUNDS LIKE A QUESTION OR INSTRUCTION. Avoid making it sound like a statement.
        NEVER CHEAT BY INCLUDING SPECIFICS ABOUT THE TEST CASES IN YOUR PROMPT. 
        ANY PROMPTS WITH THOSE SPECIFIC EXAMPLES WILL BE DISQUALIFIED.
        IF YOU USE EXAMPLES, ALWAYS USE ONES THAT ARE VERY DIFFERENT FROM THE TEST CASES.
        """
    )

    # temperature = self.temperature if i > 0 else 0 # First prompt deterministic, rest creative

    # Use ChatOpenAI for chat models
    # model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=self.temperature)  
    # messages = [
    #   SystemMessage(content=self.system_message),
    #   HumanMessage(content=prompt_content)
    # ]
    llm = model_factory.get_chat_openai()

    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = chat_prompt_template | llm | StrOutputParser()

    prompt_input_variables = {
        "description": task_description,
        "num_prompt_candidates": self.num_prompt_candidates,
        "formatted_test_cases_str": formatted_test_cases_str
    }

    response = chain.invoke(prompt_input_variables)

    prompts = self.extract_prompts(response)


    return prompts