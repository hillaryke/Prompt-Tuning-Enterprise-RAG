from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.env_loader import load_api_key

class ModelFactory:
  def __init__(self, temperature=1.5, openai_model_name="gpt-3.5-turbo", gemini_model_name="gemini-pro"):
    load_api_key()
    self.temperature = temperature
    self.openai_model_name = openai_model_name
    self.gemini_model_name = gemini_model_name

  def get_chat_openai(self):
    return ChatOpenAI(model_name=self.openai_model_name, temperature=self.temperature)

  def get_chat_gemini(self, top_p=0.85, google_api_key=None):
    return ChatGoogleGenerativeAI(
      model=self.gemini_model_name, 
      temperature=self.temperature, 
      top_p=top_p, 
      google_api_key=google_api_key
    )