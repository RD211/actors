from __future__ import annotations
import logging, random, time
from typing import Sequence
import openai
from actors.utils.logger import init_logger
from .base import LLMActor


logger = init_logger(__name__, level=logging.INFO)


class OpenAIActor(LLMActor):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        retries: int = 5,
        backoff_start: float = 1.0,
        backoff_cap: float = 30.0,
    ):
        super().__init__(name)
        openai.api_key = api_key
        openai.base_url = base_url
        self.retries = retries
        self.backoff_start = backoff_start
        self.backoff_cap = backoff_cap

    def _retry(self, fn, *args, **kwargs):
        backoff = self.backoff_start
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.verbose(f"[retry {attempt}] OpenAI error: {e}")
                if attempt == self.retries:
                    raise
                time.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, self.backoff_cap)

    def generate(self, prompts: Sequence[str], **params):
        return [
            self._retry(openai.Completion.create, model=self.name, prompt=p, **params)
            for p in prompts
        ]

    def chat(self, dialogs: Sequence[list], **params):
        return [
            self._retry(
                openai.ChatCompletion.create, model=self.name, messages=d, **params
            )
            for d in dialogs
        ]
