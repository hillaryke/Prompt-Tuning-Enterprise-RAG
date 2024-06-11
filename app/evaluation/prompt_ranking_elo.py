from app.evaluation import EloRanker
from app.misc.settings import Settings

NUMBER_OF_BATTLES = Settings.NUMBER_OF_BATTLES
SAMPLE_AMOUNT = Settings.MONTE_CARLO_SAMPLE_AMOUNT

def rank_prompts_with_elo(task_description, prompt_candidates, test_cases, retriever, num_battles=NUMBER_OF_BATTLES, sample_amount=SAMPLE_AMOUNT):
  elo_ranker = EloRanker(task_description, prompt_candidates, test_cases, retriever)
  elo_ranker.run_simulation(num_battles, sample_amount)
  ranked_prompts = elo_ranker.rank_prompts()
  return ranked_prompts