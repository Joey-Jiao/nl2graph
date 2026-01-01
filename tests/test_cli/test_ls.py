import pytest
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from nl2graph.cli import app


runner = CliRunner()


class TestLs:

    def test_ls_datasets(self):
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "data": {"metaqa": {}, "kqapro": {}},
            "data.metaqa.src": "/path/to/metaqa/src.db",
            "data.metaqa.dst": "/path/to/metaqa/dst.db",
            "data.kqapro.src": "/path/to/kqapro/src.db",
            "data.kqapro.dst": "/path/to/kqapro/dst.db",
        }.get(key, default)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.ls.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["ls", "datasets"])

        assert result.exit_code == 0
        assert "Datasets:" in result.stdout
        assert "metaqa" in result.stdout
        assert "kqapro" in result.stdout

    def test_ls_models(self):
        mock_llm_service = Mock()
        mock_llm_service.ls_providers.return_value = ["openai", "deepseek"]
        mock_llm_service.ls_models.side_effect = lambda p: {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "deepseek": ["deepseek-chat"],
        }.get(p, [])

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_llm_service

        with patch("nl2graph.cli.ls.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["ls", "models"])

        assert result.exit_code == 0
        assert "LLM Models:" in result.stdout
        assert "openai" in result.stdout
        assert "gpt-4o" in result.stdout

    def test_ls_checkpoints(self):
        mock_model_service = Mock()
        mock_model_service.ls_checkpoints.return_value = ["bart-metaqa", "bart-kqapro"]
        mock_model_service.get_checkpoint_config.side_effect = lambda n: {
            "bart-metaqa": {"path": "/models/bart-metaqa"},
            "bart-kqapro": {"path": "/models/bart-kqapro"},
        }.get(n)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_model_service

        with patch("nl2graph.cli.ls.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["ls", "checkpoints"])

        assert result.exit_code == 0
        assert "Seq2Seq Checkpoints:" in result.stdout
        assert "bart-metaqa" in result.stdout

    def test_ls_templates(self):
        mock_template_service = Mock()
        mock_template_service.ls_categories.return_value = ["prompts"]
        mock_template_service.ls_templates.return_value = ["cypher", "sparql"]

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_template_service

        with patch("nl2graph.cli.ls.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["ls", "templates"])

        assert result.exit_code == 0
        assert "Templates:" in result.stdout
        assert "prompts" in result.stdout

    def test_ls_unknown_resource(self):
        mock_ctx = Mock()

        with patch("nl2graph.cli.ls.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["ls", "unknown"])

        assert result.exit_code == 1
        assert "Unknown resource type" in result.output
