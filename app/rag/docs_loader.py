from typing import List, Tuple
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document



def load_and_split_markdown(filepath: str, splitter: List[Tuple[str, str]]):
    loader = TextLoader(filepath)
    docs = loader.load()

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=splitter)
    md_header_splits = markdown_splitter.split_text(docs[0].page_content)
    return md_header_splits
    
def load_and_split_md(DATA_PATH: str):
    documents = load_documents(DATA_PATH)
    chunks = split_text(documents)
    return chunks

def load_documents(DATA_PATH: str):
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks