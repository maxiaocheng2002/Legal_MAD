"""
Groq API client wrapper.
"""

import os
import json
from groq import Groq
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    """Wrapper for Groq API calls."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 350
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (reads from GROQ_API_KEY env if None)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment or provided")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Generate JSON response from prompt.

        Args:
            prompt: Input prompt (should request JSON output)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Parsed JSON dictionary
        """
        response_text = self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {response_text}")
            raise e


if __name__ == "__main__":
    # Test API client
    client = GroqClient()

    # Simple test
    response = client.generate("Say 'API is working' in one sentence.")
    print(f"Response: {response}")

    # JSON test
    prompt = """
    Respond in JSON format with the following structure:
    {
        "status": "success",
        "message": "API client is functional"
    }
    """
    json_response = client.generate_json(prompt)
    print(f"JSON response: {json_response}")
