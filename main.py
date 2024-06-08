# from app.rag.docs_loader import load_and_split_markdown, load_and_split_md
# from app.rag.rag_worker import rag_chain
from app.test_cases.generate_test_cases import generate_test_cases

def main():
    # answer = rag_chain.invoke("Why is the Human race odd")
    # print(answer)
    task_description = "What is few shot learning"

    test_cases = generate_test_cases(task_description, 3)
    print(test_cases)

if __name__ == '__main__':
    main()