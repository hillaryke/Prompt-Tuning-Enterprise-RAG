from app.misc.settings import Settings
from typing import List
from app.test_cases import TestCase
from app.evaluation import get_score
import random
from numpy import mean, std
from scipy.stats import norm
from rich.logging import RichHandler
import logging
from rich.console import Console

# Setup the logger
# FORMAT = "%(message)s"
# logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
console = Console()

class EloRanker:
  INITIAL_ELO_RANK = Settings.INITIAL_ELO_RANK
  SD = Settings.SD_ELO
  LEARNING_RATE = Settings.LEARNING_RATE
  BATTLE_VALUE = Settings.BATTLE_VALUE

  def __init__(self, task_description: str, prompt_candidates, test_cases: List[TestCase], retriever, model=None, embedding_model=None):
    self.task_description = task_description
    self.prompt_candidates = prompt_candidates
    self.test_cases = test_cases
    self.model = model
    self.embedding_model = embedding_model
    self.retriever = retriever
    self.elo_ranks = {prompt: self.INITIAL_ELO_RANK for prompt in self.prompt_candidates}
    self.sd = {prompt: self.SD for prompt in self.prompt_candidates}

  def expected_score(self, ratingA, ratingB):
    return 1 / (1 + 10 ** ((ratingA - ratingB) / 400))

  def update_elo(self, ratingA, ratingB, score1):
    expectedScoreA = self.expected_score(ratingB, ratingA)
    expectedScoreB = self.expected_score(ratingA, ratingB)

    roundValue = self.BATTLE_VALUE / len(self.test_cases)
    newRatingA = ratingA + roundValue * (score1 - expectedScoreA)
    newRatingB = ratingB + roundValue * (1 - score1 - expectedScoreB)

    return newRatingA, newRatingB

  def run_monte_carlo(self, sample_amount):
    distribution = {candidate: 0 for candidate in self.prompt_candidates}

    for _ in range(sample_amount):
      samples = {candidate: random.gauss(self.elo_ranks[candidate], self.sd[candidate]) for candidate in self.prompt_candidates}
      winner = max(samples, key=samples.get)
      distribution[winner] += 1

    return distribution

  def run_battle(self, candidateA, candidateB):
    console.print(f"[green]Running battle between {candidateA} and {candidateB}")

    total_score = 0
    for test_case in self.test_cases:
      score = get_score(self.task_description, test_case, candidateA, candidateB, self.retriever)  # Use the imported function here
      total_score += score

    # Calculate the average score over all test cases
    avg_score = total_score / len(self.test_cases)

    newRatingA, newRatingB = self.update_elo(self.elo_ranks[candidateA], self.elo_ranks[candidateB], avg_score)

    self.elo_ranks[candidateA] = newRatingA
    self.elo_ranks[candidateB] = newRatingB

    self.sd[candidateA] = max(self.sd[candidateA] * self.LEARNING_RATE, 125)
    self.sd[candidateB] = max(self.sd[candidateB] * self.LEARNING_RATE, 125)
    console.print(f"[blue]Battle ended. New ratings: {candidateA}: {newRatingA}, {candidateB}: {newRatingB}")


  def run_simulation(self, num_battles, sample_amount):
    for _ in range(num_battles):
      distribution = self.run_monte_carlo(sample_amount)
      candidateA = max(distribution, key=distribution.get)
      distribution[candidateA] = 0  # Exclude the first candidate from the next selection
      candidateB = max(distribution, key=distribution.get)

      self.run_battle(candidateA, candidateB)

  def rank_prompts(self):
    highest_rank = max(self.elo_ranks.values())
    ranked_prompts = [
        (
            prompt, 
            int(self.elo_ranks[prompt]), 
            int(self.elo_ranks[prompt] / highest_rank * 100)
        ) for prompt in self.prompt_candidates]
    
    ranked_prompts.sort(key=lambda x: x[1], reverse=True)
    return ranked_prompts