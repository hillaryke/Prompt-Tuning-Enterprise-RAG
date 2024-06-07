from app.rag import VectorStore

vectorstore = VectorStore()

def load_docs():
    url = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    docs = vectorstore.load_documents_from_web(url)
    return docs

def load_docs_from_web():
    docs = load_docs()
    chunks = vectorstore.split_text(docs)
    retriever =  vectorstore.save_to_chroma(chunks)
    return retriever