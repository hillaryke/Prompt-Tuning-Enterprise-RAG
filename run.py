import streamlit as st

st.title('Prompts Evaluator')

# Hardcoded ranked prompts
ranked_prompts = [
  ("Describe the principles and applications of few-shot learning.", 1060, 100),
  ("Describe the unique approach of a learning method that involves acquiring knowledge from only a limited amount of data.", 1029, 97),
  ("Please provide detailed explanations, definitions, and examples to help clarify the concept of few-shot learning.", 999, 94),
  ("Describe the principles and techniques involved in a unique approach to machine learning that aims to minimize the amount of training data required.", 969, 91),
  ("Describe the unique approach in machine learning that involves learning from a limited number of examples.", 940, 88)
]

if st.button('Generate Prompts'):
  # Display the ranked prompts along with the percentage score
  for i, (prompt, rank, percentage) in enumerate(ranked_prompts, start=1):
    st.markdown(f"### Prompt #{i}")
    st.markdown(f"**ELO Rank:** {rank} | **Percentage Score:** {percentage}%")
    st.text_area("", value=prompt, height=100, max_chars=None, key=i)