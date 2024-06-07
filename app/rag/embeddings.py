from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings

class EmbeddingFactory:
  def get_google_genai_embeddings(self, model="models/embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model)

  def get_openai_embeddings(self):
    return OpenAIEmbeddings()
  
# Embedding model example is OpenAIEmbeddings
def get_vector_embeddings(text, embedding_model = OpenAIEmbeddings()):
    """Gets the embedding for a given text."""
    return embedding_model.embed_query(text)