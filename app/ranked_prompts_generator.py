from app.test_cases import generate_test_cases
from app.rag.loaders.load_web_docs import load_docs_from_web
from app.evaluation import rank_prompts_with_elo
from app.prompts_generation import PromptGenerator

from langchain_core.output_parsers import StrOutputParser

from langchain.prompts.chat import ChatPromptTemplate
from app.utils.chat_models import ModelFactory
from app.utils.docs_utils import format_docs_to_text

ANSWER_GENERATION_TEMPERATURE = 0.7
model_factory = ModelFactory(
                    temperature=ANSWER_GENERATION_TEMPERATURE, 
                    openai_model_name="gpt-3.5-turbo"
                  )


def generate_ranked_prompts(task_description):
  # print(task_description)
  # task_description = "Write me prompts to find information about few shot learning"
  # Here we generate test_cases and prompt_candidates

  retriever = load_docs_from_web()

  test_cases = generate_test_cases(task_description, retriever, 3)

  prompt_generator = PromptGenerator(retriever)
  prompt_candidates = prompt_generator.generate_prompt_candidates(test_cases, task_description)

  ranked_prompts = rank_prompts_with_elo(task_description, prompt_candidates, test_cases, retriever)
  
  return ranked_prompts

def generate_answer(prompt: str):
    retriever = load_docs_from_web()

    prompt_template = ChatPromptTemplate.from_template(
    """
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {prompt_candidate}
        Answer:"""
    )

    context_docs = retriever.invoke(prompt)
    context_text = format_docs_to_text(context_docs)

    llm = model_factory.get_chat_openai()

    prompt_input_variables = {
        "context": context_text, 
        "prompt_candidate": prompt,
    }

    chain = prompt_template | llm | StrOutputParser()

    # Generate answer from RAG chain
    answer = chain.invoke(prompt_input_variables)
  
    return answer