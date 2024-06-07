from app.evaluation import EloRanker

def rank_prompts_with_elo(task_description, prompt_candidates, test_case, retriever):
  elo_ranker = EloRanker(task_description, prompt_candidates, test_case, retriever)
  elo_ranker.calculate_elo_ranks()
  ranked_prompts = elo_ranker.rank_prompts()
  return ranked_prompts