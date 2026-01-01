import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from transformers import AutoTokenizer

from ....base.configs import ConfigService
from .config import ConfigLoader


class Preprocessing:

    def __init__(self, config_service: ConfigService, dataset_config_path: str):
        self.config_service = config_service
        self.config_loader = ConfigLoader(dataset_config_path)
        self.dataset_config = self.config_loader.load()

        model_name = config_service.get("seq2seq.model_name_or_path", "facebook/bart-base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.dataset_config.special_tokens:
            self.tokenizer.add_tokens(self.dataset_config.special_tokens)

    def _encode_dataset(
        self,
        dataset: List[Dict],
        vocab: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inputs = [item['input'] for item in dataset]
        targets = [item['target'] for item in dataset]

        sequences = inputs + targets
        encoded = self.tokenizer(sequences, padding=True)
        max_seq_length = len(encoded['input_ids'][0])

        input_encoded = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
        )
        source_ids = np.array(input_encoded['input_ids'], dtype=np.int32)
        source_mask = np.array(input_encoded['attention_mask'], dtype=np.int32)

        target_encoded = self.tokenizer.batch_encode_plus(
            targets,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
        )
        target_ids = np.array(target_encoded['input_ids'], dtype=np.int32)

        choices = []
        answers = []
        for item in dataset:
            if 'choices' in item and 'answer' in item:
                choices.append([vocab['answer_token_to_idx'].get(w, 0) for w in item['choices']])
                answers.append(vocab['answer_token_to_idx'].get(item['answer'], 0))

        if choices:
            choices = np.array(choices, dtype=np.int32)
            answers = np.array(answers, dtype=np.int32)
        else:
            choices = np.zeros((len(dataset), 1), dtype=np.int32)
            answers = np.zeros(len(dataset), dtype=np.int32)

        return source_ids, source_mask, target_ids, choices, answers

    def process(self, input_dir: Path, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_set, val_set, test_set, vocab = self.dataset_config.load_data(input_dir)

        vocab_path = output_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)

        for name, dataset in [('train', train_set), ('val', val_set), ('test', test_set)]:
            if not dataset:
                continue

            encoded = self._encode_dataset(dataset, vocab)

            output_path = output_dir / f'{name}.pt'
            with open(output_path, 'wb') as f:
                for arr in encoded:
                    pickle.dump(arr, f)
