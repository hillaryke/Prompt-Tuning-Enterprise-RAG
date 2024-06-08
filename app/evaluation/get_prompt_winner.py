# Initialize Langchain models
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embedding_model = OpenAIEmbeddings()

# Sample Usage to Test get_score()
test_case = generated_test_cases[0]  # Choose the first test case
prompt_a = candidates[0]  # Choose the first prompt candidate
prompt_b = candidates[4]  # Choose the second prompt candidate

#Get scores
score = get_score(test_case, prompt_a, prompt_b, model, embedding_model, retriever)

print("Scores:", score)
