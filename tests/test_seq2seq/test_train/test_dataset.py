import json
import pytest
import numpy as np
import torch

from nl2graph.seq2seq.train.data.dataset import (
    load_vocab,
    collate_fn,
    Dataset,
)


class TestLoadVocab:

    def test_load_vocab(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_data = {
            "answer_token_to_idx": {"yes": 0, "no": 1},
        }
        vocab_file.write_text(json.dumps(vocab_data))

        vocab = load_vocab(vocab_file)

        assert vocab["answer_token_to_idx"] == {"yes": 0, "no": 1}
        assert vocab["answer_idx_to_token"] == {0: "yes", 1: "no"}

    def test_load_vocab_empty(self, tmp_path):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text("{}")

        vocab = load_vocab(vocab_file)

        assert vocab["answer_idx_to_token"] == {}


class TestCollate:

    def test_collate_with_targets(self):
        batch = [
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 1, 1]),
                torch.tensor([0]),
                torch.tensor([4, 5, 6]),
                torch.tensor([0]),
            ),
            (
                torch.tensor([7, 8, 9]),
                torch.tensor([1, 1, 0]),
                torch.tensor([1]),
                torch.tensor([10, 11, 12]),
                torch.tensor([1]),
            ),
        ]

        source_ids, source_mask, choices, target_ids, answer = collate_fn(batch)

        assert source_ids.shape == (2, 3)
        assert source_mask.shape == (2, 3)
        assert choices.shape == (2, 1)
        assert target_ids.shape == (2, 3)
        assert answer.shape == (2,)

    def test_collate_without_targets(self):
        batch = [
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 1, 1]),
                torch.tensor([0]),
                None,
                None,
            ),
        ]

        source_ids, source_mask, choices, target_ids, answer = collate_fn(batch)

        assert source_ids.shape == (1, 3)
        assert target_ids is None
        assert answer is None


class TestDataset:

    def test_dataset_train(self):
        source_ids = np.array([[1, 2, 3], [4, 5, 6]])
        source_mask = np.array([[1, 1, 1], [1, 1, 0]])
        target_ids = np.array([[7, 8, 9], [10, 11, 12]])
        choices = np.array([[0], [1]])
        answers = np.array([0, 1])

        dataset = Dataset((source_ids, source_mask, target_ids, choices, answers))

        assert len(dataset) == 2
        assert dataset.is_test is False

        item = dataset[0]
        assert len(item) == 5
        assert torch.equal(item[0], torch.tensor([1, 2, 3]))
        assert torch.equal(item[3], torch.tensor([7, 8, 9]))

    def test_dataset_test(self):
        source_ids = np.array([[1, 2, 3]])
        source_mask = np.array([[1, 1, 1]])
        target_ids = np.array([[7, 8, 9]])
        choices = np.array([[0]])
        answers = np.array([])

        dataset = Dataset((source_ids, source_mask, target_ids, choices, answers))

        assert dataset.is_test is True

        item = dataset[0]
        assert item[3] is None
        assert item[4] is None
