from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from .embeddings import EmbeddingFactory
import os
import bs4
import shutil

class VectorStore:
  CHROMA_PATH = "chroma"

  def __init__(self, embeddings_model=None, docs_path: str = None, docs_folder: str = None):
    self.docs_path = docs_path
    self.docs_folder = docs_folder
    self.embedding_factory = EmbeddingFactory()
    self.embeddings = self.embedding_factory.get_openai_embeddings()
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

    print(f"Saved {len(docs)} chunks to Chroma.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

  def retrieve_context(self, query, retriever=None):
    """Retrieves relevant context for a given query from your knowledge base or documents."""
    if retriever is not None:
      self.retriever = retriever
      
    if self.retriever is None:
      raise Exception("Retriever is not initialized. Call get_retriever first.")
    docs = self.retriever.invoke(query)
    return docs
  ...

  def load_and_split_docs(self):
    documents = self.load_documents(self.docs_folder)
    chunks = self.split_text(documents)
    return chunks

  def load_documents_from_dir(self, DATA_PATH: str):
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

  def load_documents_from_web(self, url: str = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"):
    loader = WebBaseLoader(
        web_paths=(url, ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    documents = loader.load()
    return documents

  def split_text(self, documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=300,
      chunk_overlap=100,
      length_function=len,
      add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
