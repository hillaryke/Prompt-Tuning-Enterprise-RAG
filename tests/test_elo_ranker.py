import unittest
from app.evaluation import EloRanker
from app.rag.loaders.load_web_docs import load_docs_from_web
from app.test_cases import TestCase

class TestEloRanker(unittest.TestCase):
  def setUp(self):
    task_description = "Some task description"
    prompt_candidates = ["Prompt 1", "Prompt 2"]
    test_cases = ["Test case 1", "Test case 2"]
    # Create TestCase instances
    test_cases = [TestCase("Test case 1", "Expected output 1"), TestCase("Test case 2", "Expected output 2")]

    retriever = load_docs_from_web()

    self.ranker = EloRanker(task_description, prompt_candidates, test_cases, retriever)

  def test_update_elo(self):
    initial_rating_A = self.ranker.elo_ranks["Prompt 1"]
    initial_rating_B = self.ranker.elo_ranks["Prompt 2"]
    new_rating_A, new_rating_B = self.ranker.update_elo(initial_rating_A, initial_rating_B, 1)

    # Add assertions here to check the new ratings

  def test_run_battle(self):
    self.ranker.run_battle("Prompt 1", "Prompt 2")

    # Add assertions here to check the updated ratings
  def test_rating_difference(self):
    initial_rating_A = self.ranker.elo_ranks["Prompt 1"]
    initial_rating_B = self.ranker.elo_ranks["Prompt 2"]
    new_rating_A, new_rating_B = self.ranker.update_elo(initial_rating_A, initial_rating_B, 1)
    self.assertNotEqual(new_rating_A, new_rating_B, "Ratings should be different after a match")

if __name__ == '__main__':
  unittest.main()