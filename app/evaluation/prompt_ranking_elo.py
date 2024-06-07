

def calculate_elo_rank(player_a_rank, player_b_rank, score, k=32):
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

# Initialize the Elo ranks of the prompts
elo_ranks = {prompt: 1000 for prompt in prompt_candidates}  # Start with a rank of 1000 for each prompt

# Calculate the new Elo ranks based on the scores
for i in range(len(prompt_candidates)):
  for j in range(i + 1, len(prompt_candidates)):
    # Get the score for the game between prompt i and prompt j
    score = get_score(test_case, prompt_candidates[i], prompt_candidates[j], model, embedding_model, retriever)
    
    # Calculate the new Elo ranks
    new_rank_i = calculate_elo_rank(elo_ranks[prompt_candidates[i]], elo_ranks[prompt_candidates[j]], score)
    new_rank_j = calculate_elo_rank(elo_ranks[prompt_candidates[j]], elo_ranks[prompt_candidates[i]], 1 - score)  # The score for prompt j is 1 - score because if prompt i wins, prompt j loses, and vice versa
    
    # Update the Elo ranks
    elo_ranks[prompt_candidates[i]] = new_rank_i
    elo_ranks[prompt_candidates[j]] = new_rank_j

print("Elo ranks:", elo_ranks)

# Get the highest rank
highest_rank = max(elo_ranks.values())

# Create a list of tuples, where each tuple is (prompt, rank, percentage score)
ranked_prompts = [(prompt, int(rank), int(rank / highest_rank * 100)) for prompt, rank in elo_ranks.items()]

# Sort the list of tuples in descending order of rank
ranked_prompts.sort(key=lambda x: x[1], reverse=True)

# Display the ranked prompts along with the percentage score
for prompt, rank, percentage in ranked_prompts:
  print(f"Prompt: {prompt}\nELO Rank: {rank}\nPercentage Score: {percentage}%\n")