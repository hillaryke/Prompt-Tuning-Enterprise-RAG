from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

class EmbeddingFactory:
  def get_google_genai_embeddings(self, model="models/embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model)

  def get_openai_embeddings(self):
    return OpenAIEmbeddings()