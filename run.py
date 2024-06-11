import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pyperclip

import time
import os
os.chdir("./")

from app.ranked_prompts_generator import generate_ranked_prompts, generate_answer

st. set_page_config(layout="wide") 

st.title('Automatic Prompts Generator')

# Create two columns
col1, col_space, col2 = st.columns([2, 0.5, 2])

# In the first column, display the prompt generation
with col1:
  # Add a text box for user input
  user_prompt = st.text_input("Enter your task or question to get ranked prompts")

  if st.button('Generate Prompts'):
    with st.spinner('Generating prompts...'):
      ranked_prompts = generate_ranked_prompts(user_prompt)
      st.session_state['ranked_prompts'] = ranked_prompts  # Store the prompts in the session state

  # Check if the prompts exist in the session state
  if 'ranked_prompts' in st.session_state:
    # Display the ranked prompts along with the percentage score
    for i, (prompt, rank, percentage) in enumerate(st.session_state['ranked_prompts'], start=1):
      markdown_header =  (
        f"<div style='background-color: rgb(37, 39, 42); padding: 10px;'>"
          f"Prompt #{i} &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;🎯&nbsp;"
            f"{percentage}% &nbsp;&nbsp;&nbsp;&nbsp;"
            f" | &nbsp;&nbsp;&nbsp;&nbsp; ELO: {rank}"
        f"</div>"
      )
      st.markdown(markdown_header, unsafe_allow_html=True)
      with stylable_container(
        "codeblock",
        """
        code {
          white-space: pre-wrap !important;
          font-family: arial;
        }
        """,
      ):
        st.code(prompt, language=None)

# In the second column, add a text box for the prompt and display the generated answer
with col2:
  # Add a text box for the prompt
  prompt = st.text_input("Enter a prompt to get an answer. (The default context is prompt engineering)")

  if st.button('Generate Answer'):
    with st.spinner('Generating answer...'):
      answer = generate_answer(prompt)  # Generate the answer for the prompt
      with stylable_container(
        "codeblock",
        """
        code {
          white-space: pre-wrap !important;
          font-family: arial;
        }
        """,
      ):
        st.code(answer, language=None)