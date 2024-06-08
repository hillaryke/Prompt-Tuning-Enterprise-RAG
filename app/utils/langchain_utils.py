from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def generate_response_with_langchain(system_message_prompt, human_message_prompt, input_variables, llm_model):
    """Generates a response using the provided prompt template and input variables."""

    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_message_prompt)
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_prompt)

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template]
    )

    chain = chat_prompt_template | llm_model | StrOutputParser()

    response = chain.invoke(input_variables)

    return response