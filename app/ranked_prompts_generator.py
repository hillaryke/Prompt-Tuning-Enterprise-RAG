from app.test_cases import TestCase, generate_test_cases
from app.rag.loaders.load_web_docs import load_docs_from_web
from app.generator import generate_answer
from app.evaluation import rank_prompts_with_elo
from app.prompts_generation import PromptGenerator

def generate_ranked_prompts(task_description):
  print(task_description)
  # task_description = "Write me prompts to find information about few shot learning"
  # Here we generate test_cases and prompt_candidates

  retriever = load_docs_from_web()

  test_cases = generate_test_cases(task_description, retriever, 3)

  prompt_generator = PromptGenerator(retriever)
  prompt_candidates = prompt_generator.generate_prompt_candidates(test_cases, task_description)

  ranked_prompts = rank_prompts_with_elo(task_description, prompt_candidates, test_cases, retriever)
  
  return ranked_prompts