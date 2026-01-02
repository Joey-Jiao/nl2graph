import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from nl2graph.cli import app


runner = CliRunner()


@pytest.fixture
def temp_data_dir(tmp_path):
    data_file = tmp_path / "test.json"
    data_file.write_text(json.dumps([
        {"id": "q001", "question": "What is X?", "answer": ["A"]},
        {"id": "q002", "question": "What is Y?", "answer": ["B"]},
    ]))
    return tmp_path, data_file


@pytest.fixture
def mock_config(temp_data_dir):
    tmp_path, data_file = temp_data_dir
    config = Mock()

    def get_side_effect(key, default=None):
        mapping = {
            "data.test.eval.data": str(data_file),
            "data.test.src": str(tmp_path / "src.db"),
            "data.test.dst": str(tmp_path / "dst.db"),
        }
        return mapping.get(key, default)

    config.get = Mock(side_effect=get_side_effect)
    return config


class TestInit:

    def test_init_success(self, mock_config, temp_data_dir):
        tmp_path, _ = temp_data_dir

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["init", "test"])

        assert result.exit_code == 0
        assert "Loaded 2 records" in result.stdout
        assert (tmp_path / "src.db").exists()
        assert (tmp_path / "dst.db").exists()

    def test_init_no_data_path(self):
        mock_config = Mock()
        mock_config.get.return_value = None

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["init", "unknown"])

        assert result.exit_code == 1
        assert "No data path configured" in result.output

    def test_init_data_file_not_found(self, tmp_path):
        mock_config = Mock()

        def get_side_effect(key, default=None):
            mapping = {
                "data.test.eval.data": str(tmp_path / "nonexistent.json"),
            }
            return mapping.get(key, default)

        mock_config.get = Mock(side_effect=get_side_effect)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["init", "test"])

        assert result.exit_code == 1
        assert "Data file not found" in result.output

    def test_init_with_json_override(self, tmp_path):
        data_file = tmp_path / "custom.json"
        data_file.write_text(json.dumps([
            {"id": "q001", "question": "Q1", "answer": ["A1"]},
        ]))

        mock_config = Mock()

        def get_side_effect(key, default=None):
            mapping = {
                "data.test.src": str(tmp_path / "src.db"),
                "data.test.dst": str(tmp_path / "dst.db"),
            }
            return mapping.get(key, default)

        mock_config.get = Mock(side_effect=get_side_effect)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["init", "test", "--json", str(data_file)])

        assert result.exit_code == 0
        assert "Loaded 1 records" in result.stdout

    def test_init_idempotent(self, mock_config, temp_data_dir):
        tmp_path, _ = temp_data_dir

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result1 = runner.invoke(app, ["init", "test"])
            assert result1.exit_code == 0

            result2 = runner.invoke(app, ["init", "test"])
            assert result2.exit_code == 0
            assert "Removed existing" in result2.stdout

        from nl2graph.data.repository import SourceRepository
        with SourceRepository(str(tmp_path / "src.db")) as src:
            assert src.count() == 2


class TestClear:

    def test_clear_gen_cascades(self, mock_config, temp_data_dir):
        tmp_path, data_file = temp_data_dir

        from nl2graph.data.repository import SourceRepository, ResultRepository
        from nl2graph.data.entity import GenerationResult, ExecutionResult, EvaluationResult

        with SourceRepository(str(tmp_path / "src.db")) as src:
            src.init_from_json(str(data_file))

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            gen = GenerationResult(query="MATCH (n) RETURN n")
            exec_ = ExecutionResult(result=["A"], success=True)
            eval_ = EvaluationResult(exact_match=1.0, f1=1.0, precision=1.0, recall=1.0)
            dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
            dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec_)
            dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval_)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.clear.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["clear", "test", "-m", "llm", "--model", "gpt-4o", "-l", "cypher"])

        assert result.exit_code == 0
        assert "Cleared 1 records" in result.stdout

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            r = dst.get("q001", "llm", "cypher", "gpt-4o")
            assert r.gen is None
            assert r.exec is None
            assert r.eval is None

    def test_clear_exec_cascades(self, mock_config, temp_data_dir):
        tmp_path, data_file = temp_data_dir

        from nl2graph.data.repository import SourceRepository, ResultRepository
        from nl2graph.data.entity import GenerationResult, ExecutionResult, EvaluationResult

        with SourceRepository(str(tmp_path / "src.db")) as src:
            src.init_from_json(str(data_file))

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            gen = GenerationResult(query="MATCH (n) RETURN n")
            exec_ = ExecutionResult(result=["A"], success=True)
            eval_ = EvaluationResult(exact_match=1.0, f1=1.0, precision=1.0, recall=1.0)
            dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
            dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec_)
            dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval_)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.clear.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["clear", "test", "-m", "llm", "--model", "gpt-4o", "-l", "cypher", "-s", "exec"])

        assert result.exit_code == 0

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            r = dst.get("q001", "llm", "cypher", "gpt-4o")
            assert r.gen is not None
            assert r.exec is None
            assert r.eval is None

    def test_clear_eval_only(self, mock_config, temp_data_dir):
        tmp_path, data_file = temp_data_dir

        from nl2graph.data.repository import SourceRepository, ResultRepository
        from nl2graph.data.entity import GenerationResult, ExecutionResult, EvaluationResult

        with SourceRepository(str(tmp_path / "src.db")) as src:
            src.init_from_json(str(data_file))

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            gen = GenerationResult(query="MATCH (n) RETURN n")
            exec_ = ExecutionResult(result=["A"], success=True)
            eval_ = EvaluationResult(exact_match=1.0, f1=1.0, precision=1.0, recall=1.0)
            dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)
            dst.save_execution("q001", "llm", "cypher", "gpt-4o", exec_)
            dst.save_evaluation("q001", "llm", "cypher", "gpt-4o", eval_)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.clear.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["clear", "test", "-m", "llm", "--model", "gpt-4o", "-l", "cypher", "-s", "eval"])

        assert result.exit_code == 0

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            r = dst.get("q001", "llm", "cypher", "gpt-4o")
            assert r.gen is not None
            assert r.exec is not None
            assert r.eval is None

    def test_clear_dst_not_found(self, tmp_path):
        mock_config = Mock()

        def get_side_effect(key, default=None):
            if key == "data.test.dst":
                return str(tmp_path / "nonexistent.db")
            return default

        mock_config.get = Mock(side_effect=get_side_effect)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.clear.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["clear", "test", "-m", "llm", "--model", "gpt-4o", "-l", "cypher"])

        assert result.exit_code == 1
        assert "dst.db not found" in result.output


class TestExport:

    def test_export_success(self, mock_config, temp_data_dir):
        tmp_path, data_file = temp_data_dir

        from nl2graph.data.repository import SourceRepository, ResultRepository
        from nl2graph.data.entity import GenerationResult

        with SourceRepository(str(tmp_path / "src.db")) as src:
            src.init_from_json(str(data_file))

        with ResultRepository(str(tmp_path / "dst.db")) as dst:
            gen = GenerationResult(query="MATCH (n) RETURN n")
            dst.save_generation("q001", "llm", "cypher", "gpt-4o", gen)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        output_path = tmp_path / "output.json"

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["export", "test", "--output", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()

    def test_export_dst_not_configured(self):
        mock_config = Mock()
        mock_config.get.return_value = None

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["export", "unknown"])

        assert result.exit_code == 1
        assert "No dst.db path configured" in result.output

    def test_export_dst_not_found(self, tmp_path):
        mock_config = Mock()

        def get_side_effect(key, default=None):
            if key == "data.test.dst":
                return str(tmp_path / "nonexistent.db")
            return default

        mock_config.get = Mock(side_effect=get_side_effect)

        mock_ctx = Mock()
        mock_ctx.resolve.return_value = mock_config

        with patch("nl2graph.cli.init.get_context", return_value=mock_ctx):
            result = runner.invoke(app, ["export", "test"])

        assert result.exit_code == 1
        assert "dst.db not found" in result.output
