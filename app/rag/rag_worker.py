import dotenv
from operator import itemgetter

from app.rag.docs_loader import load_and_split_markdown, load_and_split_md
from app.rag.chroma_vectorstore import get_retriever
from app.misc import Settings

from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate

dotenv.load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

system_message_prompt = SystemMessagePromptTemplate.from_template(Settings.PROMT_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def format_docs(context_docs):
    return "\n\n---\n\n".join([doc.page_content for doc, _score in context_docs])

# docs = load_and_split_markdown('data/docs/bank_name_docs.md', Settings.HEADERS_TO_SPLIT)
# ensemble_retriever = get_retriever(
#     docs, 
#     Settings.BM25_K, Settings.MMR_K, Settings.MMR_FETCH_K, 
#     Settings.METADATA_INFO, Settings.CONTENT_DESCRIPTION
# )

docs = load_and_split_md('data/books')

retriever = get_retriever(docs)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# rag_chain_with_source = RunnableParallel(
#     {"documents": ensemble_retriever, "question": RunnablePassthrough()}
# ) | {
#     "documents": lambda input: [doc.metadata for doc in input["documents"]],
#     "answer": rag_chain_from_docs,
# }
