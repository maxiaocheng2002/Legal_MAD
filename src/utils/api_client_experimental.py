"""
Groq and OpenRouter API client wrappers.
"""

import os
import json
import time
import requests
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
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_format: Optional[str] = None,
        include_reasoning: Optional[bool] = None
    ) -> str:
        """
        Generate text from prompt with retry logic for rate limits.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            response_format: Optional response format (e.g., {"type": "json_object"})
            reasoning_format: Optional reasoning format (raw, parsed, hidden)
            include_reasoning: Whether to include reasoning in response (mutually exclusive with reasoning_format)

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

        if reasoning_format:
            kwargs["reasoning_format"] = reasoning_format

        if include_reasoning is not None:
            kwargs["include_reasoning"] = include_reasoning

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
        Falls back to non-JSON mode if response_format is not supported.

        Args:
            prompt: Input prompt (should request JSON output)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Parsed JSON dictionary
        """
        # Try with JSON mode first
        try:
            response_text = self.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                #include_reasoning=False  # Disable reasoning to save tokens
            )
        except Exception as e:
            # If JSON mode fails, try without it but disable reasoning to save tokens
            error_str = str(e).lower()
            if "json_validate_failed" in error_str or "400" in error_str:
                print(f"⚠️  JSON mode failed for {self.model}, retrying without JSON mode and with include_reasoning=False...")
                response_text = self.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    include_reasoning=False  # Disable reasoning to avoid token waste
                    #reasoning_format="parsed"  
                    # No response_format
                )
            else:
                raise

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {response_text}")
            raise e


class OpenRouterClient:
    """Wrapper for OpenRouter API calls."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.0,
        max_tokens: int = 500,
        max_retries: int = 10,
        retry_delay: float = 2.0,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (reads from OPENROUTER_API_KEY env if None)
            model: Model name (default: llama-3.3-70b-instruct:free)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries for rate limit errors
            retry_delay: Initial delay between retries (exponential backoff)
            site_url: Optional URL for leaderboard attribution
            site_name: Optional name for leaderboard attribution
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or provided")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.site_url = site_url
        self.site_name = site_name

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Optional attribution headers for leaderboard
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        return headers

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
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

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens,
            "effort": "minimal"
        }

        if response_format:
            payload["response_format"] = response_format
        

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60
                )

                # Check for rate limit (429) or server errors (5xx)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"Rate limit/Server error (status {response.status_code}). "
                              f"Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()

                # Raise for other HTTP errors (including 400 Bad Request)
                # Don't retry 400 errors as they indicate invalid request parameters
                response.raise_for_status()

                # Parse response
                data = response.json()

                # Debug: Print response structure
                message = data["choices"][0]["message"]
                print(f"DEBUG - Message structure: {message}")

                # Handle different response formats from OpenRouter
                if isinstance(message, dict):
                    # If message has 'content' field, use it
                    if "content" in message:
                        content = message["content"]
                        # If content is itself a dict with 'content', extract it
                        if isinstance(content, dict) and "content" in content:
                            return content["content"]

                        # Strip markdown code blocks if present
                        if isinstance(content, str) and content.strip().startswith('```'):
                            content = content.strip()
                            # Remove opening ```json or ```
                            if content.startswith('```json'):
                                content = content[7:]  # Remove ```json
                            elif content.startswith('```'):
                                content = content[3:]   # Remove ```
                            # Remove closing ```
                            if content.endswith('```'):
                                content = content[:-3]
                            content = content.strip()

                        return content if content else ""
                    # If no 'content', try to find it in nested structure
                    elif "role" in message and message["role"] == "assistant":
                        # Qwen reasoning model case - content might be empty
                        return message.get("content", "")

                # Fallback: return the whole message as string
                return str(message)



            except requests.exceptions.HTTPError:
                # Re-raise HTTPError immediately (including 400 Bad Request)
                # These are not retriable errors
                raise
            except requests.exceptions.RequestException as e:
                # Only retry on network/timeout errors, not HTTP errors
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Network error: {e}. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        effort: Optional[str] = None
    ) -> Dict:
        """
        Generate JSON response from prompt.
        Falls back to non-JSON mode if response_format is not supported.

        Args:
            prompt: Input prompt (should request JSON output)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Parsed JSON dictionary
        """
        # Try with JSON mode first
        try:
            response_text = self.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except requests.exceptions.HTTPError as e:
            # If 400 Bad Request, likely JSON mode not supported - retry without it
            if e.response.status_code == 400:
                print(f"⚠️  JSON mode not supported by this model, retrying without response_format...")
                response_text = self.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=None
                )
            else:
                raise

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {response_text}")
            raise e


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING GROQ CLIENT")
    print("=" * 60)

    # Test Groq client
    groq_client = GroqClient()

    # Simple test
    print("\n1. Testing basic generation...")
    response = groq_client.generate("Say 'Groq API is working' in one sentence.")
    print(f"   Response: {response}")

    # JSON test
    print("\n2. Testing JSON generation...")
    prompt = """
    Respond in JSON format with the following structure:
    {
        "status": "success",
        "message": "Groq client is functional"
    }
    """
    json_response = groq_client.generate_json(prompt)
    print(f"   JSON response: {json_response}")

    print("\n" + "=" * 60)
    print("TESTING OPENROUTER CLIENT")
    print("=" * 60)

    # Test OpenRouter client
    try:
        or_client = OpenRouterClient()

        print("\n1. Testing basic generation...")
        response = or_client.generate("Say 'OpenRouter API is working' in one sentence.")
        print(f"   Response: {response}")

        print("\n2. Testing JSON generation...")
        prompt = """
        Respond in JSON format with the following structure:
        {
            "status": "success",
            "message": "OpenRouter client is functional"
        }
        """
        json_response = or_client.generate_json(prompt)
        print(f"   JSON response: {json_response}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except ValueError as e:
        print(f"\n⚠️  OpenRouter test skipped: {e}")
        print("   Make sure OPENROUTER_API_KEY is set in .env file")
    except Exception as e:
        print(f"\n❌ OpenRouter test failed: {e}")
