import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from nl2graph.base.configs import ConfigService
from nl2graph.generation.seq2seq.train.preprocessing import Preprocessing


class TestPreprocessing:

    @pytest.fixture
    def mock_config_service(self):
        config_service = Mock(spec=ConfigService)
        config_service.get.side_effect = lambda key, default=None: {
            "seq2seq.model_name_or_path": "facebook/bart-base",
        }.get(key, default)
        return config_service

    @pytest.fixture
    def sample_config(self, tmp_path):
        config_file = tmp_path / "config.py"
        config_file.write_text("""
special_tokens = []

def load_data(input_dir):
    import json
    from pathlib import Path
    input_dir = Path(input_dir)
    train_set = json.load(open(input_dir / 'train.json'))
    val_set = json.load(open(input_dir / 'val.json'))
    test_set = json.load(open(input_dir / 'test.json'))

    for dataset in [train_set, val_set, test_set]:
        for item in dataset:
            item['input'] = item.get('question', '')
            item['target'] = item.get('LF', '')

    vocab = {'answer_token_to_idx': {}}
    return train_set, val_set, test_set, vocab
""")
        return config_file

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        train_data = [
            {"question": "What is X?", "LF": "MATCH (n:X) RETURN n"},
            {"question": "Who is Y?", "LF": "MATCH (n:Y) RETURN n"},
        ]
        val_data = [
            {"question": "Where is Z?", "LF": "MATCH (n:Z) RETURN n"},
        ]
        test_data = [
            {"question": "When is A?", "LF": "MATCH (n:A) RETURN n"},
        ]

        (data_dir / "train.json").write_text(json.dumps(train_data))
        (data_dir / "val.json").write_text(json.dumps(val_data))
        (data_dir / "test.json").write_text(json.dumps(test_data))

        return data_dir

    @patch("nl2graph.generation.seq2seq.train.preprocessing.AutoTokenizer")
    def test_init(self, mock_tokenizer_class, mock_config_service, sample_config):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        preprocessing = Preprocessing(mock_config_service, str(sample_config))

        assert preprocessing.dataset_config is not None
        assert preprocessing.dataset_config.special_tokens == []
        mock_tokenizer_class.from_pretrained.assert_called_once()

    @patch("nl2graph.generation.seq2seq.train.preprocessing.AutoTokenizer")
    def test_process(self, mock_tokenizer_class, mock_config_service, sample_config, sample_data_dir, tmp_path):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_tokenizer.batch_encode_plus.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        preprocessing = Preprocessing(mock_config_service, str(sample_config))
        output_dir = tmp_path / "output"

        preprocessing.process(sample_data_dir, output_dir)

        assert (output_dir / "train.pt").exists()
        assert (output_dir / "val.pt").exists()
        assert (output_dir / "test.pt").exists()
        assert (output_dir / "vocab.json").exists()

        vocab = json.loads((output_dir / "vocab.json").read_text())
        assert "answer_token_to_idx" in vocab
