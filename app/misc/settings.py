from typing import Final
from langchain.chains.query_constructor.base import AttributeInfo

class Settings:
    HEADERS_TO_SPLIT: Final = [
        ("#", "Topic Header"),
        ("##", "Subtopic Header"),
        ("###", "Paragraph Header")
    ]

    METADATA_INFO: Final = [
        AttributeInfo(
            name="Title",
            description="Part of the document where the text was taken from",
            type="string or list[string]",
        ),
    ]

    CONTENT_DESCRIPTION: Final = "Description of banking products"

    PROMT_TEMPLATE: Final = """
    You are an assistant who answers user questions.
    Use fragments of the received context to answer the question.
    If you don't know the answer, say that you don't know, don't make up an answer.
    Use a maximum of three sentences and be concise.\n
    Question: {question} \n
    Context: {context} \n
    Answer:
    """

    BM25_K: Final = 2
    MMR_K: Final = 2
    MMR_FETCH_K: Final = 5

