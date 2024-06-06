from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
# DATA_PATH = "data/books"

def get_retriever(docs):
    return save_to_chroma(docs)

def save_to_chroma(docs: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())