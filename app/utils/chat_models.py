from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class ModelFactory:
    def __init__(self, temperature=1.5):
      self.temperature = temperature

    def get_chat_openai(self, model_name="gpt-3.5-turbo"):
        return ChatOpenAI(model_name=model_name, temperature=self.temperature)

    # If there is no environment variable set for the API key, you can pass the API
    # key to the parameter `google_api_key` of the `ChatGoogleGenerativeAI` function:
    # `google_api_key="key"`.
    def get_chat_gemini(self, model="gemini-pro", top_p=0.85, google_api_key=None):
        return ChatGoogleGenerativeAI(
                    model=model, 
                    temperature=self.temperature, 
                    top_p=top_p, 
                    google_api_key=google_api_key
        )