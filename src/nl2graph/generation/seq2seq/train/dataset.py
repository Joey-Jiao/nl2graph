import io
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler


def load_vocab(path: Path) -> Dict:
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = {v: k for k, v in vocab.get('answer_token_to_idx', {}).items()}
    return vocab


def collate_fn(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[-1][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    return source_ids, source_mask, choices, target_ids, answer


class Dataset(TorchDataset):

    def __init__(self, inputs: Tuple):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers = inputs
        self.is_test = len(self.answers) == 0 or self.answers is None

    def __getitem__(self, index: int):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index]) if self.choices is not None else torch.LongTensor([0])

        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])

        return source_ids, source_mask, choices, target_ids, answer

    def __len__(self) -> int:
        return len(self.source_ids)


class DataLoader(TorchDataLoader):

    def __init__(
        self,
        vocab_path: Path,
        data_path: Path,
        batch_size: int,
        training: bool = False,
    ):
        self.vocab = load_vocab(vocab_path)

        with open(data_path, 'rb') as f:
            file_content = f.read()
        buffer = io.BytesIO(file_content)

        inputs = []
        for _ in range(5):
            obj = pickle.load(buffer)
            inputs.append(obj)

        dataset = Dataset(tuple(inputs))

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )


class DistributedDataLoader(TorchDataLoader):

    def __init__(
        self,
        dataset: Dataset,
        vocab: Dict,
        batch_size: int,
        sampler: DistributedSampler,
    ):
        self.vocab = vocab
        self.sampler = sampler

        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def prepare_dataset(vocab_path: Path, data_path: Path) -> Tuple[Dataset, Dict]:
    vocab = load_vocab(vocab_path)

    with open(data_path, 'rb') as f:
        file_content = f.read()
    buffer = io.BytesIO(file_content)

    inputs = []
    for _ in range(5):
        obj = pickle.load(buffer)
        inputs.append(obj)

    dataset = Dataset(tuple(inputs))
    return dataset, vocab
