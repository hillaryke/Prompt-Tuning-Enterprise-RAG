import numpy as np
from py import test
from app.misc import Settings
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.test_cases import TestCase
from app.test_cases.generate_test_cases import TEMPERATURE
from app.utils.chat_models import ModelFactory
from app.rag.embeddings import get_vector_embeddings
from app.generator import generate_answer
from app.rag.embeddings import EmbeddingFactory
from langchain_community.embeddings import OpenAIEmbeddings

TEMPERATURE = Settings.TEMPERATURE_GET_PROMPT_WINNER
model_factory = ModelFactory(TEMPERATURE)
llm = model_factory.get_chat_openai()
llm_gemini = model_factory.get_chat_gemini()

embedding_model = EmbeddingFactory().get_openai_embeddings()

# TODO - Here we should use gpt-4 or we can try with Gemini
# Define the cache as a dictionary outside the function
cache = {}

def get_score(task_description, test_case, prompt_a, prompt_b, retreiver, embedding_model = None, model = None):
    if embedding_model is None:
        embedding_model = embedding_model

    # Generate a unique key for the pair of prompts
    key = (task_description, test_case, prompt_a, prompt_b)

    # If the result is in the cache, return it
    if key in cache:
        return cache[key]

    """
        Calculates the score for a prompt comparison using either using LLM or embedding similarity.
        Returns 1 if prompt A is better, 0 if prompt B is better, and 0.5 if they are equally good.
    """
    answer_a = generate_answer(prompt_a, test_case, retreiver)
    answer_b = generate_answer(prompt_b, test_case, retreiver)

    if test_case.expected_output.strip():
        print("USING LLM to compare prompts")
        score = comparePromptsUsingLLM(task_description, test_case, answer_a, answer_b)
    else:  # Use embeddings to calculate similarity
        print("USING COSINE to compare prompts")
        score = comparePromptsUsingCosineSimilarity(test_case, embedding_model=OpenAIEmbeddings())

    # Store the score in the cache before returning it
    cache[key] = score
    return score

def comparePromptsUsingLLM(task_description: str, test_case: TestCase, answer_a: str, answer_b: str):
    system_message_prompt = SystemMessagePromptTemplate.from_template(Settings.RANKING_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
        Task: {task_description}
        Prompt: {test_case_scenario}
        Expected output of the Prompt: {test_case_expected_output}
        Generation A: {answer_a}
        Generation B: {answer_b}
        """
    )

    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = chat_prompt_template | llm | StrOutputParser()

    prompt_input_variables = {
        "task_description": task_description, 
        "test_case_scenario": test_case.scenario,
        "test_case_expected_output": test_case.expected_output,
        "answer_a": answer_a, 
        "answer_b": answer_b
    }

    winner = chain.invoke(prompt_input_variables)

    while winner not in ['A', 'B', 'DRAW']:
        print("Invalid input from the model. Please try generating again.")

        winner = chain.invoke(prompt_input_variables)
    return 1 if winner == 'A' else 0 if winner == 'B' else 0.5 


def comparePromptsUsingCosineSimilarity(answer_a, answer_b, test_case, embedding_model):
    embedding_a = get_vector_embeddings(answer_a, embedding_model)
    embedding_b = get_vector_embeddings(answer_b, embedding_model)
    embedding_expected = get_vector_embeddings(test_case.expected_output, embedding_model)

    score_a = cosine_similarity(embedding_a, embedding_expected)
    score_b = cosine_similarity(embedding_b, embedding_expected)

    # Handle ties and near-ties
    if abs(score_a - score_b) < 0.1:  # Adjust threshold as needed
        return 0.5  # Draw

    return 1 if score_a > score_b else 0  # Return 1 if A wins, 0 if B wins


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
        Calculates the cosine similarity between two vector embeddings a and b.
        Returns a float value between -1 and 1.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))