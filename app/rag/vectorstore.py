from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from embeddings import EmbeddingFactory
import os
import shutil

class VectorStore:
  CHROMA_PATH = "chroma"

  def __init__(self, docs_path, embeddings_model="models/embedding-001"):
    self.docs_path = docs_path
    self.embedding_factory = EmbeddingFactory()
    self.embeddings = self.embedding_factory.get_google_genai_embeddings(model=embeddings_model)
    self.retriever = None

  def get_retriever(self, docs):
    self.retriever = self.save_to_chroma(docs)
    return self.retriever

  def save_to_chroma(self, docs: list[Document]):
    # Clear out the database first.
    if os.path.exists(self.CHROMA_PATH):
      shutil.rmtree(self.CHROMA_PATH)

    # Create a new DB from the documents.
    vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

  def retrieve_context(self, query):
    """Retrieves relevant context for a given query from your knowledge base or documents."""
    if self.retriever is None:
      raise Exception("Retriever is not initialized. Call get_retriever first.")
    docs = self.retriever.invoke(query)
    return docs
  ...

  def load_and_split_md(self):
    documents = self.load_documents(self.docs_path)
    chunks = self.split_text(documents)
    return chunks

  def load_documents(self, DATA_PATH: str):
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

  def split_text(self, documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=200,
      chunk_overlap=100,
      length_function=len,
      add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
