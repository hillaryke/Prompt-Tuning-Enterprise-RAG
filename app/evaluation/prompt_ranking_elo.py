from app.evaluation import EloRanker

def rank_prompts_with_elo(task_description, prompt_candidates, test_cases, retriever, num_battles=1000, sample_amount=1000):
  elo_ranker = EloRanker(task_description, prompt_candidates, test_cases, retriever)
  elo_ranker.run_simulation(num_battles, sample_amount)
  ranked_prompts = elo_ranker.rank_prompts()
  return ranked_prompts