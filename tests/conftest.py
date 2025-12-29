import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_record_data():
    return {
        "question": "What is the capital of France?",
        "answer": ["Paris"],
        "hop": 1,
    }


@pytest.fixture
def sample_run_result_data():
    return {
        "method": "llm",
        "lang": "cypher",
        "model": "gpt-4o",
    }


@pytest.fixture
def temp_config_dir(tmp_path):
    config_file = tmp_path / "configs.yaml"
    config_file.write_text("""
log:
  log_dir: "outputs/logs"
  console_level: "INFO"

llm:
  openai:
    gpt-4o:
      timeout: 60

templates:
  prompts: "templates/prompts"

seq2seq:
  pretrained_dir: "models/pretrained"
  models:
    bart-base:
      path: "models/pretrained/bart-base"
      max_length: 512
      special_tokens: []
""")
    return tmp_path


@pytest.fixture
def temp_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key\nDEEPSEEK_API_KEY=test-key-2\n")
    return env_file


@pytest.fixture
def temp_template_dir(tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)
    template_file = prompts_dir / "test.jinja2"
    template_file.write_text("Question: {{ question }}\nLang: {{ lang }}")
    return tmp_path
