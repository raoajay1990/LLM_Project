from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    """
    Abstract base class for all LLM wrappers.
    Any new model/provider we add later must inherit from this class 
    and implement these methods.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Takes a list of message dictionaries (role & content) and optionally a list of tools.
        Must return the LLM's response.
        """
        pass