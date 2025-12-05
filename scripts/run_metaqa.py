import argparse

from nl2graph.base.context import get_context
from nl2graph.pipeline.service import PipelineService

DATASET = "metaqa"
ALL_SUBSETS = ["1-hop", "2-hop", "3-hop"]
ALL_SPLITS = ["train", "dev", "test"]


def run_single(pipeline, runner, schema, subset, split, mode):
    run_id = runner.run_id
    processed_path = pipeline.get_data_path(DATASET, "processed", subset, split)
    result_path = pipeline.get_data_path(DATASET, "result", subset, split)

    try:
        records = pipeline.load_records(result_path)
    except FileNotFoundError:
        records = pipeline.load_records(processed_path)

    if mode == "generate":
        records = runner.generate(records, schema)
        for record in records:
            if run_id in record.runs:
                record.runs[run_id].exec = None
                record.runs[run_id].eval = None
    elif mode == "execute":
        records = runner.execute(records)
        for record in records:
            if run_id in record.runs:
                record.runs[run_id].eval = None
    elif mode == "evaluate":
        records = runner.evaluate(records)

    pipeline.save_records(records, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "execute", "evaluate"])
    parser.add_argument("--subset", type=str, default=None, choices=ALL_SUBSETS)
    parser.add_argument("--split", type=str, default=None, choices=ALL_SPLITS)
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--lang", type=str, default="cypher")
    parser.add_argument("--template", type=str, default="nl2cypher.jinja2")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    ctx = get_context()
    pipeline = ctx.resolve(PipelineService)
    schema = pipeline.load_schema(DATASET)

    need_executor = args.mode in ["execute", "evaluate"]
    runner = pipeline.create_runner(
        provider=args.provider,
        model=args.model,
        lang=args.lang,
        prompt_template=args.template,
        dataset=DATASET if need_executor else None,
        workers=args.workers,
    )

    subsets = [args.subset] if args.subset else ALL_SUBSETS
    splits = [args.split] if args.split else ALL_SPLITS

    for subset in subsets:
        for split in splits:
            run_single(pipeline, runner, schema, subset, split, args.mode)


if __name__ == "__main__":
    main()
