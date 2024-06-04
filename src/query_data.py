import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from pprint import pprint

from datasets import Dataset
from ragas.metrics import faithfulness
from ragas.metrics.critique import harmfulness
from ragas import evaluate

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme({
    "info": "green",
    "warning": "yellow",
    "salmon": "light_salmon3",
    "danger": "bold red",
    "plum": "plum2",
    "success": "bold green",
})
console = Console(theme=custom_theme)


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI. - Parse the query text from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB. - Load the Chroma database
    # We also need the embedding function which is the same as we used to create the database.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB - for the chunk that best matches our query

    """
    The return type of the similarity_search_with_relevance_scores method is a list of tuples.
    Each tuple contains a Document object and a relevance score. - List[Tuple[Document, float]]
    """
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # check if there are not matches or first results is below a certain threshold, return early
    # This ensures we only proceed if we have a good match.
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    print("==============================================")
    console.print(context_text, style="info")


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    console.print(prompt, style="warning")

    model = ChatOpenAI()
    # response_text = model.predict(prompt)
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)

    data = {
        'question': [query_text],
        'answer': [response_text],
        'contexts': [[doc.page_content for doc, _score in results]]
    }

    console.print(data, style="plum")

    dataset = Dataset.from_dict(data)
    score = evaluate(dataset, metrics=[faithfulness, harmfulness])
    score.to_pandas()

    console.print(score, style="success")


if __name__ == "__main__":
    main()