import json
from pathlib import Path
from typing import List, Optional

from ..base.configs import ConfigService
from ..base.llm.service import LLMService
from ..base.templates.service import TemplateService
from ..graph.service import GraphService
from ..graph.schema.base import BaseSchema
from ..graph.schema.property_graph import PropertyGraphSchema
from .entity import Record
from .generator import QueryGenerator
from .executor import QueryExecutor
from .evaluator import Evaluator
from .runner import PipelineRunner


class PipelineService:
    def __init__(
        self,
        config: ConfigService,
        llm_service: LLMService,
        template_service: TemplateService,
        graph_service: GraphService,
    ):
        self.config = config
        self.llm_service = llm_service
        self.template_service = template_service
        self.graph_service = graph_service

    def load_records(self, path: str) -> List[Record]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [Record(**item) for item in data]

    def save_records(self, records: List[Record], path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = [record.model_dump(exclude_none=True) for record in records]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_runner(
        self,
        provider: str,
        model: str,
        lang: str,
        prompt_template: str,
        dataset: Optional[str] = None,
        workers: int = 1,
    ) -> PipelineRunner:
        generator = QueryGenerator(
            llm_service=self.llm_service,
            template_service=self.template_service,
            provider=provider,
            model=model,
            lang=lang,
            prompt_template=prompt_template,
        )

        executor = None
        if dataset:
            connector = self.graph_service.get_connector(dataset, lang)
            executor = QueryExecutor(connector)

        evaluator = Evaluator()

        return PipelineRunner(
            generator=generator,
            executor=executor,
            evaluator=evaluator,
            lang=lang,
            model=f"{provider}/{model}",
            workers=workers,
        )

    def get_data_path(self, dataset: str, stage: str, subset: str, split: str) -> str:
        base = self.config.get(f"data.{dataset}.{stage}")
        if not base:
            raise KeyError(f"data path not found: data.{dataset}.{stage}")
        return str(Path(base) / subset / f"{split}.json")

    def load_schema(self, dataset: str) -> BaseSchema:
        schema_path = self.config.get(f"data.{dataset}.schema")
        if not schema_path:
            raise KeyError(f"schema path not found: data.{dataset}.schema")

        path = Path(schema_path)
        if not path.exists():
            raise FileNotFoundError(f"schema file not found: {schema_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PropertyGraphSchema.from_dict(data)

    def aggregate_metrics(self, records: List[Record], run_id: str) -> dict:
        metrics = {
            "total": len(records),
            "gen_success": 0,
            "exec_success": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "hits_at_1": 0.0,
        }

        eval_count = 0
        for record in records:
            run = record.runs.get(run_id)
            if not run:
                continue
            if run.gen and run.gen.query_processed:
                metrics["gen_success"] += 1
            if run.exec and run.exec.success:
                metrics["exec_success"] += 1
            if run.eval:
                eval_count += 1
                metrics["exact_match"] += run.eval.exact_match or 0.0
                metrics["f1"] += run.eval.f1 or 0.0
                metrics["precision"] += run.eval.precision or 0.0
                metrics["recall"] += run.eval.recall or 0.0
                metrics["hits_at_1"] += run.eval.hits_at_1 or 0.0

        if eval_count > 0:
            metrics["exact_match"] /= eval_count
            metrics["f1"] /= eval_count
            metrics["precision"] /= eval_count
            metrics["recall"] /= eval_count
            metrics["hits_at_1"] /= eval_count

        return metrics
