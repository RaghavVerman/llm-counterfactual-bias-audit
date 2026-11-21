"""Batched, seeded text generation for base and instruction-tuned causal LMs."""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_INSTRUCTION = (
    "Continue the following text with one or two sentences. "
    "Reply with only the continuation, without repeating the text.\n\n{prompt}"
)


class Generator:
    """Wraps a Hugging Face causal LM.

    Base models (e.g. GPT-2) receive the prompt verbatim and we keep the
    continuation. Models that ship a chat template (e.g. Qwen2.5-Instruct) get
    the prompt inside an instruction, and the assistant reply is the
    continuation. Either way the output is "what the model said about the
    person", so the same scorers apply.
    """

    def __init__(self, model_name_or_path: str, device: str | None = None):
        self.name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"  # required for batched decoder-only generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.is_chat = bool(getattr(self.tokenizer, "chat_template", None))

    def _format(self, prompt: str) -> str:
        if not self.is_chat:
            return prompt
        messages = [{"role": "user", "content": CHAT_INSTRUCTION.format(prompt=prompt)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        seed: int,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        torch.manual_seed(seed)
        texts = [self._format(p) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            add_special_tokens=not self.is_chat,
        ).to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [clean_continuation(d) for d in decoded]


_SENT_END = re.compile(r"(?<=[.!?])\s")


def clean_continuation(text: str, max_sentences: int = 2) -> str:
    """Keep the first paragraph and at most ``max_sentences`` sentences.

    Base LMs drift off-topic after a sentence or two; scoring the drift would
    measure noise rather than how the model describes the person.
    """
    text = text.strip().split("\n\n")[0].replace("\n", " ").strip()
    parts = _SENT_END.split(text)
    return " ".join(parts[:max_sentences]).strip()
