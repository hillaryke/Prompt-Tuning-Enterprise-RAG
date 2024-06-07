from app.rag.docs_loader import load_and_split_markdown, load_and_split_md
from app.rag.rag_worker import rag_chain

def main():
    answer = rag_chain.invoke("Why is the Human race odd")
    print(answer)

if __name__ == '__main__':
    main()