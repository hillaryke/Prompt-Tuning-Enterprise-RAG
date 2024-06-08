from typing import Final
from langchain.chains.query_constructor.base import AttributeInfo

class Settings:
    TEMPERATURE_GET_PROMPT_WINNER: Final = 0

    NUMBER_OF_PROMPT_CANDIDATES: Final = 3

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

    CREATE_TEST_CASES_SYSTEM_PROMPT = """
            Your job is to create a test case for a given task and it's expected output given the context. The task is a description of a use-case.

            DO NOT GET OUT OF CONTEXT GIVEN. ONLY STICK WITHIN THE CONTEXT. DO NOT MAKE UP ANYTHING OUTSIDE OF THE CONTEXT.

            It should be general enough that it can be used to test the AI's ability to perform the task in retreiving documents.
            It should never actually complete the task, but it should be a good example of the task.

            Example:
            Task: Creates a landing page headline for a new product
            Test case: "A new type of toothpaste that whitens teeth in 5 minutes"
            Test case: "A fitness app that helps you lose weight"
            Test case: "A therapist for dogs"

            Task: Generates a title for a blog post that will get the most clicks
            Test case: "How to increase your productivity by 10x"
            Test case: "A post about the best travel destinations in the world"
            Test case: "The best restaurants in New York"

            Task: Generate a paragraph that describes a product
            Test case: "The new Macbook Pro"
            Test case: "Nike shoes"
            Test case: "A case for iPhones that's velvety smooth and very durable"

            You will be graded based on the performance of your test case and expected output... but don't cheat! You cannot include specifics about the task in your test case. Any test cases with examples will be disqualified.
            Be really creative! The most creative test cases will be rewarded.

            YOU NEVER OUTPUT SOMETHING THAT COMPLETES THE TASK. ONLY A TEST CASE AND ITS EXPECTED OUTPUT.

            Most importantly, output NOTHING but the test case and expected output. Do not include anything else in your message.
            Each test case should include:
            * Scenario: A clear description of the situation or input to be tested.
            * Expected output: The ideal or expected output from the system.

            Format each test case like this:
            Scenario: [Scenario description]
            Expected output: [Expected output]
    """


    PROMPT_CANDIDATES_GENERATION_SYSTEM_PROMPT = """  
        You are an AI language model with expertise in natural language processing and information retrieval. 
        Utilize your semantic understanding and query analysis capabilities to generate superior prompts for automatic prompt generation systems. 
        Your goal is to optimize user queries, enhance prompt relevance, and improve overall retrieval accuracy.
        
        Given the task description and test cases and the context, generate the asked number of prompts that are diverse, informative, and contextually relevant.

        ENSURE YOU FOLLOW THIS FORMAT EXACTLY WHEN RETURNING PROMPTS. START WITH 'Prompt:' AND THEN THE PROMPT. 
        DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE EXCEPT THE PROMPTS.:
        Prompt: <Generated Prompt 1>
        Prompt: <Generated Prompt 2>
        Prompt: <Generated Prompt 3>
        ...
        
        The generated prompt must satisfy the rules given below:
        0. The generated prompted should only contain the prompt and no numbering
        1.The prompt should make sense to humans even when read without the given context.
        2.The prompt should be fully created from the given context.
        3.The prompt should be framed from a part of context that contains important information. It can also be from tables,code,etc.
        4.The prompt must be reasonable and must be understood and responded by humans.
        5.Do no use phrases like 'provided context',etc in the prompt
        6.The prompt should not contain more than 15 words, make of use of abbreviation wherever possible.
        7.The prompt should not be a verbatim copy of the context.
        8.The prompt should not include double quotes, instead just given it as is.
        9.The prompt should not include any grammatical errors.

        Respond with the prompts, and nothing else. Be creative.
        ENSURE THE PROMPT SOUNDS LIKE A QUESTION OR INSTRUCTION. Avoid making it sound like a statement.
        NEVER CHEAT BY INCLUDING SPECIFICS ABOUT THE TEST CASES IN YOUR PROMPT. 
        ANY PROMPTS WITH THOSE SPECIFIC EXAMPLES WILL BE DISQUALIFIED.
        IF YOU USE EXAMPLES, ALWAYS USE ONES THAT ARE VERY DIFFERENT FROM THE TEST CASES.
    """

    RANKING_PROMPT = """
        Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.

        You will be provided with the task description, the test prompt, and two generations - one for each system prompt.

        Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.

        Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.

        Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

        Respond with your ranking, and nothing else. Be fair and unbiased in your judgement.
    """

