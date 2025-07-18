from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Sequence

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

    async def _aretry(self, afn, *args, **kwargs):
        backoff = self.backoff_start
        for attempt in range(1, self.retries + 1):
            try:
                return await afn(*args, **kwargs)
            except Exception as e:
                logger.verbose(f"[async retry {attempt}] OpenAI error: {e}")
                if attempt == self.retries:
                    raise
                await asyncio.sleep(backoff + random.uniform(0, 1))
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

    async def agenerate(self, prompts: Sequence[str], **params):
        tasks = [
            self._aretry(openai.Completion.acreate, model=self.name, prompt=p, **params)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    async def achat(self, dialogs: Sequence[list], **params):
        tasks = [
            self._aretry(
                openai.ChatCompletion.acreate, model=self.name, messages=d, **params
            )
            for d in dialogs
        ]
        return await asyncio.gather(*tasks)
