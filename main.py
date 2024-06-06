from app.rag.docs_loader import load_and_split_markdown, load_and_split_md

def main():
    # docs = load_and_split_markdown('data/docs/bank_name_docs.md', [('##', '###')])
    docs = load_and_split_md('data/books')
    print(docs[:5])
    return docs

if __name__ == '__main__':
    main()