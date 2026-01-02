import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from typer.testing import CliRunner

from nl2graph.cli import app


runner = CliRunner()


@pytest.fixture
def temp_db_setup(tmp_path):
    data_file = tmp_path / "test.json"
    data_file.write_text(json.dumps([
        {"id": "q001", "question": "What is X?", "answer": ["A"]},
    ]))

    from nl2graph.data.repository import SourceRepository, ResultRepository
    with SourceRepository(str(tmp_path / "src.db")) as src:
        src.init_from_json(str(data_file))
    with ResultRepository(str(tmp_path / "dst.db")) as dst:
        pass

    return tmp_path


class TestGenerate:

    def test_generate_src_not_found(self, tmp_path):
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data.test.src": str(tmp_path / "nonexistent.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
        }.get(key, default)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.generate.get_context", return_value=mock_ctx):
            result = runner.invoke(app, [
                "generate", "test",
                "--method", "llm",
                "--model", "gpt-4o",
                "--lang", "cypher",
            ])

        assert result.exit_code == 1
        assert "src.db not found" in result.output

    def test_generate_unknown_method(self, temp_db_setup):
        tmp_path = temp_db_setup

        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data.test.src": str(tmp_path / "src.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
        }.get(key, default)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.generate.get_context", return_value=mock_ctx):
            result = runner.invoke(app, [
                "generate", "test",
                "--method", "unknown",
                "--model", "test-model",
                "--lang", "cypher",
            ])

        assert result.exit_code == 1
        assert "Unknown method" in result.output

    def test_generate_unknown_llm_provider(self, temp_db_setup):
        tmp_path = temp_db_setup

        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data.test.src": str(tmp_path / "src.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
        }.get(key, default)

        mock_llm_service = Mock()

        mock_ctx = Mock()
        mock_ctx.resolve.side_effect = lambda cls: {
            "ConfigService": mock_config,
            "LLMService": mock_llm_service,
        }.get(cls.__name__, mock_config)

        with patch("nl2graph.cli.generate.get_context", return_value=mock_ctx):
            result = runner.invoke(app, [
                "generate", "test",
                "--method", "llm",
                "--model", "unknown-model",
                "--lang", "cypher",
            ])

        assert result.exit_code == 1
        assert "Unknown model provider" in result.output

    def test_generate_llm_success(self, temp_db_setup):
        tmp_path = temp_db_setup

        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data.test.src": str(tmp_path / "src.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
            "data.test.schema": str(tmp_path / "schema.json"),
        }.get(key, default)

        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({
            "name": "test",
            "entities": [{"label": "Node", "properties": {"name": "str"}}],
            "relations": [],
        }))

        mock_llm_service = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "MATCH (n) RETURN n"
        mock_client.chat.return_value = mock_response
        mock_llm_service.get_client.return_value = mock_client

        mock_template_service = Mock()
        mock_template_service.ls_templates.return_value = ["cypher", "sparql"]
        mock_template_service.render.return_value = "prompt with schema"

        mock_ctx = Mock()
        mock_ctx.resolve.side_effect = lambda cls: {
            "ConfigService": mock_config,
            "LLMService": mock_llm_service,
            "TemplateService": mock_template_service,
        }.get(cls.__name__, mock_config)

        with patch("nl2graph.cli.generate.get_context", return_value=mock_ctx):
            result = runner.invoke(app, [
                "generate", "test",
                "--method", "llm",
                "--model", "gpt-4o",
                "--lang", "cypher",
            ])

        assert result.exit_code == 0
        assert "Generating for 1 records" in result.output
        assert "Done" in result.output

    def test_generate_seq2seq_checkpoint_not_found(self, temp_db_setup):
        tmp_path = temp_db_setup

        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data.test.src": str(tmp_path / "src.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
        }.get(key, default)

        mock_model_service = Mock()
        mock_model_service.get_checkpoint_config.return_value = None

        mock_ctx = Mock()
        mock_ctx.resolve.side_effect = lambda cls: {
            "ConfigService": mock_config,
            "ModelService": mock_model_service,
        }.get(cls.__name__, mock_config)

        with patch("nl2graph.cli.generate.get_context", return_value=mock_ctx):
            result = runner.invoke(app, [
                "generate", "test",
                "--method", "seq2seq",
                "--model", "nonexistent",
                "--lang", "cypher",
            ])

        assert result.exit_code == 1
        assert "Checkpoint" in result.output and "not found" in result.output
