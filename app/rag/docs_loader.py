from typing import List, Tuple
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document



def load_and_split_markdown(filepath: str, splitter: List[Tuple[str, str]]):
    loader = TextLoader(filepath)
    docs = loader.load()

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=splitter)
    md_header_splits = markdown_splitter.split_text(docs[0].page_content)
    return md_header_splits
    