"""
LLM client wrapper built on top of OpenAI's Python SDK with retry logic.
"""
import os
from typing import Dict, List, Optional

import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model_name: Target model name.
            api_key: Optional API key; falls back to env.
            base_url: Optional custom endpoint (e.g., proxy, compatible provider).
        """
        model_lower = (model_name or "").lower()

        # Resolve API key
        resolved_api_key = api_key or os.getenv("api_key")
        if "llama-guard" in model_lower or "openrouter" in model_lower:
            resolved_api_key = resolved_api_key or os.getenv("OPENROUTER_API_KEY")
        elif "deepseek" in model_lower:
            resolved_api_key = resolved_api_key or os.getenv("DEEPSEEK_API_KEY")
        else:
            resolved_api_key = resolved_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("API key must be provided via argument or env (OPENAI_API_KEY / DEEPSEEK_API_KEY / api_key).")

        # Resolve base url
        resolved_base_url = base_url
        if resolved_base_url is None:
            if "llama-guard" in model_lower or "openrouter" in model_lower:
                resolved_base_url = (
                    os.getenv("OPENROUTER_BASE_URL")
                    or "https://openrouter.ai/api/v1"
                )
            elif "deepseek" in model_lower:
                resolved_base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("base_url")
            else:
                resolved_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("base_url")

        self.model_name = model_name
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        reraise=True,
    )
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        """
        Call the chat completion endpoint with retries on rate-limit/connection issues.
        """
        request_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            request_args["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_args)
        return response.choices[0].message.content or ""

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Retrieve vector embedding for given text.
        """
        response = self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def one_shot_query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        """
        Convenience helper to send a single-turn prompt.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


if __name__ == "__main__":
    client = LLMClient()
    try:
        reply = client.one_shot_query(prompt="Say hello", system_prompt="You are a friendly assistant.")
        print(f"LLMClient test reply: {reply}")
    except Exception as exc:
        print(f"LLMClient test failed: {exc}")
