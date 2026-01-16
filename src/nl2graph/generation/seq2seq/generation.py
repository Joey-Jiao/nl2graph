from pathlib import Path
from typing import List, Optional, Any

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from ...base import ConfigService
from ...data import GenerationOutput
from ...execution.schema.base import BaseSchema


class Generation:

    def __init__(
        self,
        model_path: str,
        config_service: Optional[ConfigService] = None,
        tokenizer_path: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        device: Optional[str] = None,
        translator: Optional[Any] = None,
        lang: str = "cypher",
    ):
        self.max_length = 512
        if config_service:
            self.max_length = config_service.get("seq2seq.max_length", 512)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = translator
        self.lang = lang

        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if special_tokens:
            self.tokenizer.add_tokens(special_tokens)

        model_path = Path(model_path)
        if model_path.exists():
            self.model = BartForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True
            )
        else:
            self.model = BartForConditionalGeneration.from_pretrained(str(model_path))

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str, schema: Optional[BaseSchema] = None) -> GenerationOutput:
        encoded = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
            )

        content = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if self.translator:
            content = self._translate_ir(content)

        return GenerationOutput(content=content)

    def _translate_ir(self, ir: str) -> str:
        try:
            if self.lang == "cypher":
                return self.translator.to_cypher(ir)
            elif self.lang == "sparql":
                return self.translator.to_sparql(ir)
            elif self.lang == "kopl":
                return self.translator.to_kopl(ir)
        except Exception:
            return ir
        return ir
