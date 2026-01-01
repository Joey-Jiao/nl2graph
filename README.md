# nl2graph

NL-to-Graph Query research framework for evaluating natural language to graph database query methods.

## Usage

```
nl2graph
├── init <dataset>                    Initialize src.db and dst.db
│   └── [-j, --json <path>]           Override data.json path
│
├── generate <dataset>                Generate queries from questions
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language: cypher, sparql, kopl (required)
│   ├── [-t, --template <name>]       Prompt template name
│   ├── [--ir]                        Enable IR mode (seq2seq)
│   ├── [--hop <n>]                   Filter by hop
│   ├── [--split <name>]              Filter by split
│   └── [-w, --workers <n>]           Parallel workers (default: 1)
│
├── execute <dataset>                 Execute queries against database
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   ├── [--hop <n>]                   Filter by hop
│   ├── [--split <name>]              Filter by split
│   └── [-w, --workers <n>]           Parallel workers (default: 1)
│
├── evaluate <dataset>                Evaluate execution results
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   ├── [--hop <n>]                   Filter by hop
│   └── [--split <name>]              Filter by split
│
├── train <dataset>                   Train seq2seq model
│   ├── [-s, --shot <1shot|3shot|5shot>]  Few-shot config
│   ├── [-f, --from <checkpoint>]     Transfer from checkpoint
│   ├── [-p, --preprocess-only]       Only run preprocessing
│   └── [-o, --output <path>]         Output directory
│
├── report <dataset>                  Generate evaluation report
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   ├── [-f, --format <json|markdown>]  Output format (default: json)
│   └── [-o, --output <path>]         Output file path
│
├── export <dataset>                  Export dst.db to JSON
│   └── [-o, --output <path>]         Output JSON path
│
└── ls <resource>                     List resources
    └── resource: datasets | models | checkpoints | templates
```

## Examples

```bash
nl2graph init metaqa
nl2graph generate metaqa -m llm --model gpt-4o -l cypher -t cypher
nl2graph execute metaqa -m llm --model gpt-4o -l cypher
nl2graph evaluate metaqa -m llm --model gpt-4o -l cypher
nl2graph report metaqa -m llm --model gpt-4o -l cypher
```

## Data Flow

```
src.db (read-only)                    dst.db (read-write)
┌─────────────────┐                   ┌─────────────────────────────────────┐
│ Record          │                   │ Result                              │
│  - id           │                   │  - record_id ─┐                     │
│  - question     │──── generate ────►│  - method    ─┼─ composite key      │
│  - answer       │                   │  - lang      ─┤                     │
│  - hop          │                   │  - model     ─┘                     │
│  - split        │                   │  - gen:  {query_raw, query, ir}     │
└─────────────────┘                   │  - exec: {success, result, error}   │
        │                             │  - eval: {exact_match, f1, ...}     │
        │                             └─────────────────────────────────────┘
        │                                       │
        └───────────── evaluate ────────────────┘
                   (compare answers)
```

### Database Schema

**src.db** (read-only) - Source records from dataset

| Table: `records` | Type | Description |
|------------------|------|-------------|
| `id` | TEXT | Primary key |
| `question` | TEXT | Natural language question |
| `answer` | JSON | Ground truth answer(s) |
| `hop` | INTEGER | (optional) Query complexity |
| `split` | TEXT | (optional) train/dev/test |
| `...` | | Additional fields allowed |

**dst.db** (read-write) - Experiment results

| Table: `results` | Type | Description |
|------------------|------|-------------|
| `record_id` | TEXT | ┐ |
| `method` | TEXT | │ Composite |
| `lang` | TEXT | │ Primary Key |
| `model` | TEXT | ┘ |
| `gen` | JSON | GenerationResult |
| `exec` | JSON | ExecutionResult |
| `eval` | JSON | EvaluationResult |

The composite key `(record_id, method, lang, model)` allows multiple experiment results for the same record.
