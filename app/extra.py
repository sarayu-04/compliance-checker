# %%
from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory
import langfuse

class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )

class Synthesizer:
    def __init__(self):
        # Initialize Langfuse client
        self.langfuse_client = langfuse.Client(api_key="your_langfuse_api_key")

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content", "category"]
        )

        # Fetch the prompt from Langfuse
        prompt_id = "your_prompt_id"  # Replace with your actual prompt ID
        system_prompt = Synthesizer.fetch_prompt_from_langfuse(prompt_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)

    @staticmethod
    def fetch_prompt_from_langfuse(prompt_id: str) -> str:
        """
        Fetch the prompt from Langfuse using the prompt ID.

        Args:
            prompt_id (str): The ID of the prompt to fetch.

        Returns:
            str: The prompt text.
        """
        # Initialize Langfuse client (if not already initialized)
        langfuse_client = langfuse.Client(api_key="your_langfuse_api_key")

        # Fetch the prompt
        prompt = langfuse_client.get_prompt(prompt_id)

        # Return the prompt text
        return prompt.text

# %%
