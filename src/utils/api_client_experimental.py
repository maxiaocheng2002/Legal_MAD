"""
Groq API client wrapper.
"""

import os
import json
import time
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
        max_tokens: int = 500,  # Experimental version uses higher token limit
        max_retries: int = 5,
        retry_delay: float = 2.0
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (reads from GROQ_API_KEY env if None)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries for rate limit errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment or provided")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text from prompt with retry logic for rate limits.

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

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if 'rate_limit' in error_str or 'rate limit' in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit exceeded after {self.max_retries} retries.")
                        raise
                else:
                    # Not a rate limit error, raise immediately
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("Max retries exceeded")

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
