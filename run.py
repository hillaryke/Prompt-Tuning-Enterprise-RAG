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


# Add a text box for user input
user_prompt = st.text_input("Enter you task description or question to get alternative ranked prompts")

if st.button('Generate Prompts'):
  # Display the ranked prompts along with the percentage score
  for i, (prompt, rank, percentage) in enumerate(ranked_prompts, start=1):
    markdown_header =  (
        f"<div style='background-color: rgb(37, 39, 42); padding: 10px;'>"
            f"Prompt #{i} &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;🎯&nbsp;"
                  f"{percentage}% &nbsp;&nbsp;&nbsp;&nbsp;"
                  f" | &nbsp;&nbsp;&nbsp;&nbsp; ELO: {rank}"
        f"</div>"
    )
    st.markdown(markdown_header, unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: rgb(70, 72, 74); padding: 10px;'>{prompt}</div>", unsafe_allow_html=True)