from app.misc import Settings
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.test_cases import TestCase
from app.test_cases.generate_test_cases import TEMPERATURE
from app.utils.chat_models import ModelFactory

TEMPERATURE = Settings.TEMPERATURE_GET_PROMPT_WINNER
model_factory = ModelFactory(TEMPERATURE)
llm = model_factory.get_chat_openai()

# TODO - Here we should use gpt-4 or we can try with Gemini
def comparePromptsUsingLLM(task_description: str, test_case_scenario: str, answer_a: str, answer_b: str):
    system_message_prompt = SystemMessagePromptTemplate.from_template(Settings.RANKING_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
        Task: {task_description}
        Prompt: {test_case_scenario}
        Generation A: {answer_a}
        Generation B: {answer_b}
        """
    )

    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = chat_prompt_template | llm | StrOutputParser()

    prompt_input_variables = {
        "task_description": task_description, 
        "test_case_scenario": test_case_scenario, 
        "answer_a": answer_a, 
        "answer_b": answer_b
    }

    winner = chain.invoke(prompt_input_variables)

    while winner not in ['A', 'B', 'DRAW']:
        print("Invalid input from the model. Please try generating again.")

        winner = chain.invoke(prompt_input_variables)
    return 1 if winner == 'A' else 0 if winner == 'B' else 0.5 

