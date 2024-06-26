from langchain_core.output_parsers import StrOutputParser

from langchain.prompts.chat import ChatPromptTemplate
from app.utils.chat_models import ModelFactory
from app.rag.vectorstore import VectorStore
from app.test_cases import TestCase
from app.utils.docs_utils import format_docs_to_text

ANSWER_GENERATION_TEMPERATURE = 0.7
model_factory = ModelFactory(
                    temperature=ANSWER_GENERATION_TEMPERATURE, 
                    openai_model_name="gpt-3.5-turbo"
                  )

def generate_answer(prompt_candidate: str, test_case: TestCase, retriever, model = None):
    """Generates a response using the provided prompt and test case."""

    prompt_template = ChatPromptTemplate.from_template(
    """
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {prompt_candidate}
        Scenario: {scenario}
        Answer:"""
    )

    context_docs = retriever.invoke(test_case.scenario)
    context_text = format_docs_to_text(context_docs)

    llm = model_factory.get_chat_openai()

    prompt_input_variables = {
        "context": context_text, 
        "prompt_candidate": prompt_candidate, 
        "scenario": test_case.scenario
    }

    chain = prompt_template | llm | StrOutputParser()

    # Generate answer from RAG chain
    answer = chain.invoke(prompt_input_variables)

    return answer
