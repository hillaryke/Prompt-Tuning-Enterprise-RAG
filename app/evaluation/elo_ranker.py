from app.test_cases import TestCase
from app.evaluation import get_score

class EloRanker:
  INITIAL_ELO_RANK = 1000
  K_FACTOR = 32

  def __init__(self, task_description: str, prompt_candidates, test_case: TestCase, retriever, model=None, embedding_model=None):
    self.task_description = task_description
    self.prompt_candidates = prompt_candidates
    self.test_case = test_case
    self.model = model
    self.embedding_model = embedding_model
    self.retriever = retriever
    self.elo_ranks = {prompt: self.INITIAL_ELO_RANK for prompt in self.prompt_candidates}

  def calculate_elo_rank(self, player_a_rank, player_b_rank, score, k=32):
    """
    Calculate the new Elo rank of a player based on the score of a game.
    
    player_a_rank: The current Elo rank of player A
    player_b_rank: The current Elo rank of player B
    score: The score of the game (1 for a win, 0.5 for a draw, 0 for a loss)
    k: The K-factor, which determines the maximum change in rank (default: 32)
    """
    # Calculate the expected score of player A
    expected_score_a = 1 / (1 + 10 ** ((player_b_rank - player_a_rank) / 400))
    
    # Update the Elo rank of player A
    new_rank_a = player_a_rank + k * (score - expected_score_a)
    
    return new_rank_a

  def calculate_elo_ranks(self):
    for i in range(len(self.prompt_candidates)):
      for j in range(i + 1, len(self.prompt_candidates)):
        score = get_score(
                    self.task_description,
                    self.test_case, 
                    self.prompt_candidates[i],
                    self.prompt_candidates[j],
                    self.retriever
                  )
        new_rank_i = self.calculate_elo_rank(self.elo_ranks[self.prompt_candidates[i]], self.elo_ranks[self.prompt_candidates[j]], score)
        new_rank_j = self.calculate_elo_rank(self.elo_ranks[self.prompt_candidates[j]], self.elo_ranks[self.prompt_candidates[i]], 1 - score)
        self.elo_ranks[self.prompt_candidates[i]] = new_rank_i
        self.elo_ranks[self.prompt_candidates[j]] = new_rank_j

  def rank_prompts(self):
    highest_rank = max(self.elo_ranks.values())
    ranked_prompts = [(prompt, int(rank), int(rank / highest_rank * 100)) for prompt, rank in self.elo_ranks.items()]
    ranked_prompts.sort(key=lambda x: x[1], reverse=True)
    return ranked_prompts
