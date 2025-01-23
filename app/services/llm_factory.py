from typing import Any, Dict, List, Type
from pydantic import BaseModel
import google.generativeai as genai
import os
from config.settings import get_settings

class LLMFactory:
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self.settings = get_settings().gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        # Combine messages into a single prompt
        prompt = self._format_messages(messages)
        
        # Generate response using Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.settings.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.settings.max_tokens),
            )
        )

        # Parse the response into the expected format
        result = self._parse_response(response, response_model)
        return result

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt string."""
        formatted_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        return "\n\n".join(formatted_messages)

    def _parse_response(self, response: Any, response_model: Type[BaseModel]) -> BaseModel:
        """Parse Gemini response into the expected Pydantic model."""
        response_text = response.text
        
        # If it's a SynthesizedResponse, create the proper structure
        if response_model.__name__ == 'SynthesizedResponse':
            return response_model(
                answer=response_text,
                thought_process=["Analyzed context", "Generated response using Gemini model"],
                enough_context=True  # You might want to make this more dynamic based on context
            )
        
        # For other response types
        return response_model(**{"content": response_text})
