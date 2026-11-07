"""Real Responses client for the earlier multi-agent project's generate API."""
from __future__ import annotations

import fse2027_llm_client as transport
import uuid


class OpenCodeGoClient:
    # Deliberately expose generate(), so the old runtime uses this interface
    # rather than injecting its obsolete gpt-4 Chat Completions defaults.
    def __init__(self, config: dict):
        if config.get('base_url', transport.BASE_URL).rstrip('/') != transport.BASE_URL:
            raise ValueError('Only the official OpenCode Go endpoint is enabled')
        if config.get('model', transport.MODEL) != transport.MODEL:
            raise ValueError('The configured model must be gpt-5.6-luna')
        self.model = transport.MODEL
        self.max_tokens = int(config.get('max_tokens', 2000))
        self.last_record = None
        self.session_id = str(uuid.uuid4())

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat_with_history([{'role': 'user', 'content': prompt}], **kwargs)

    def chat_with_history(self, messages: list, model=None, temperature=None,
                          max_tokens=None, **kwargs) -> str:
        if model is not None and model != self.model:
            raise ValueError('Model overrides must remain gpt-5.6-luna')
        expected_temperature = transport.settings().get('temperature', 0)
        if temperature is not None and temperature != expected_temperature:
            raise ValueError('Temperature override differs from the shared OpenCode configuration')
        if kwargs:
            raise ValueError('Unsupported OpenCode request options: ' + ', '.join(sorted(kwargs)))
        self.last_record = transport.request_chat(messages, session_id=self.session_id,
            max_tokens=(self.max_tokens if max_tokens is None else int(max_tokens)))
        return self.last_record['response']['choices'][0]['message']['content']
