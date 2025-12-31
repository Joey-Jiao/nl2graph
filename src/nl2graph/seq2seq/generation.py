from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from ..base.configs import ConfigService


class Generation:

    def __init__(
        self,
        model_path: str,
        config_service: Optional[ConfigService] = None,
        tokenizer_path: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.max_length = 512
        self.batch_size = 32
        if config_service:
            self.max_length = config_service.get("seq2seq.max_length", 512)
            self.batch_size = config_service.get("seq2seq.inference.batch_size", 32)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    def generate(self, text: str, max_length: Optional[int] = None) -> str:
        max_length = max_length or self.max_length
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        max_length = max_length or self.max_length
        batch_size = batch_size or self.batch_size
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                )

            batch_predictions = [
                self.tokenizer.decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for output in outputs
            ]
            predictions.extend(batch_predictions)

        return predictions
