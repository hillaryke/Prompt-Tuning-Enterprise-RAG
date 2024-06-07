from app.misc import Settings
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.test_cases import TestCase
from app.test_cases.generate_test_cases import TEMPERATURE
from app.utils.chat_models import ModelFactory
from app.rag.embeddings import get_vector_embeddings

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

def comparePromptsUsingCosineSimilarity(test_case, embedding_model):
    embedding_a = get_vector_embeddings(answer_a, embedding_model)
    embedding_b = get_vector_embeddings(answer_b, embedding_model)
    embedding_expected = get_vector_embeddings(test_case.expected_output, embedding_model)

    score_a = cosine_similarity(embedding_a, embedding_expected)
    score_b = cosine_similarity(embedding_b, embedding_expected)

    # Handle ties and near-ties
    if abs(score_a - score_b) < 0.1:  # Adjust threshold as needed
        return 0.5  # Draw

    return 1 if score_a > score_b else 0  # Return 1 if A wins, 0 if B wins