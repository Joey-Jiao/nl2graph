import pytest
from pathlib import Path

from nl2graph.generation.seq2seq.train.config import ConfigLoader, DatasetConfig


class TestConfigLoader:

    @pytest.fixture
    def sample_config(self, tmp_path):
        config_file = tmp_path / "config.py"
        config_file.write_text("""
special_tokens = ['<a>', '<b>']

def load_data(args):
    return [], [], [], {}

def evaluate(args, outputs, targets, *xargs):
    return 0.0

def translate(args, outputs, targets):
    return outputs, targets
""")
        return config_file

    @pytest.fixture
    def minimal_config(self, tmp_path):
        config_file = tmp_path / "minimal.py"
        config_file.write_text("""
special_tokens = []

def load_data(args):
    return [], [], [], {}
""")
        return config_file

    def test_load_config(self, sample_config):
        loader = ConfigLoader(str(sample_config))
        config = loader.load()

        assert isinstance(config, DatasetConfig)
        assert config.special_tokens == ['<a>', '<b>']
        assert callable(config.load_data)
        assert callable(config.evaluate)
        assert callable(config.translate)

    def test_load_minimal_config(self, minimal_config):
        loader = ConfigLoader(str(minimal_config))
        config = loader.load()

        assert config.special_tokens == []
        assert callable(config.load_data)
        assert config.evaluate is None
        assert config.translate is None

    def test_load_nonexistent_config(self, tmp_path):
        loader = ConfigLoader(str(tmp_path / "nonexistent.py"))

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_module_accessible(self, sample_config):
        loader = ConfigLoader(str(sample_config))
        loader.load()

        assert loader.module is not None
        assert hasattr(loader.module, 'special_tokens')
