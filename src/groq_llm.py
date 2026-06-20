import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Import our blueprint
from .llm_base import BaseLLM

class GroqLLM(BaseLLM):
    """
    Groq-specific implementation of our LLM blueprint.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        # Call the parent class constructor
        super().__init__(model_name)

        # Initialize the Groq client. 
        # It automatically looks for GROQ_API_KEY in your environment variables.
        # self.client = Groq(api_key=os.getenv("GROQ_API_KEY"),base_url="https://api.groq.com/openai/v1")
        load_dotenv()
        self.client = OpenAI(api_key= os.getenv("GROQ_API_KEY"), base_url=os.getenv("GROQ_BASE_URL"))

    def generate_response(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Calls the Groq API. Handles both standard chat and tool-calling.
        """
        # We set up a dictionary of arguments to pass to the API
        api_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        # If tools are provided, we add them to our arguments
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"

        # Make the actual API call using unpacking (**api_kwargs)
        response = self.client.chat.completions.create(**api_kwargs)

        return response
    