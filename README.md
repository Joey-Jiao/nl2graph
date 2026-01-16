# nl2graph

NL-to-Graph Query research framework for evaluating natural language to graph database query methods.

## Usage

```
nl2graph
├── init <dataset>                    Initialize src.db and dst.db (idempotent, removes existing)
│   └── [-j, --json <path>]           Override data.json path
│
├── generate <dataset>                Generate queries from questions
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language: cypher, sparql, gremlin (required)
│   ├── [--ir]                        Enable IR mode (seq2seq)
│   ├── [--hop <n>]                   Filter by hop
│   ├── [--split <name>]              Filter by split
│   ├── [-w, --workers <n>]           Parallel workers (default: 1)
│   └── [--if-exists <skip|override>] Action when record exists (default: skip)
│
├── execute <dataset>                 Execute queries against database
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   ├── [--hop <n>]                   Filter by hop
│   ├── [--split <name>]              Filter by split
│   ├── [-w, --workers <n>]           Parallel workers (default: 1)
│   └── [--if-exists <skip|override>] Action when record exists (default: skip)
│
├── evaluate <dataset>                Evaluate execution results
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   ├── [--hop <n>]                   Filter by hop
│   ├── [--split <name>]              Filter by split
│   └── [--if-exists <skip|override>] Action when record exists (default: skip)
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
├── clear <dataset>                   Clear results from dst.db
│   ├── -m, --method <llm|seq2seq>    Generation method (required)
│   ├── --model <name>                Model name (required)
│   ├── -l, --lang <lang>             Query language (required)
│   └── [-s, --stage <gen|exec|eval>] Stage to clear (default: gen, cascades)
│
├── server                            Manage graph database servers
│   ├── start <dataset>               Start server (docker-compose up)
│   │   ├── -l, --lang <lang>         Query language (required)
│   │   └── [-t, --timeout <sec>]     Startup timeout (default: 60)
│   ├── stop <dataset>                Stop server (docker-compose down)
│   │   └── -l, --lang <lang>         Query language (required)
│   └── status <dataset>              Check server status
│       └── -l, --lang <lang>         Query language (required)
│
└── ls <resource>                     List resources
    └── resource: datasets | models | checkpoints | templates
```

## Examples

```bash
nl2graph init metaqa
nl2graph generate metaqa -m llm --model gpt-4o -l cypher
nl2graph execute metaqa -m llm --model gpt-4o -l cypher
nl2graph evaluate metaqa -m llm --model gpt-4o -l cypher
nl2graph report metaqa -m llm --model gpt-4o -l cypher
```

## Data Flow

```
src.db (read-only)                    dst.db (read-write)
┌─────────────────┐                   ┌─────────────────────────────────────┐
│ Record          │                   │ Result                              │
│  - id           │                   │  - question_id ─┐                   │
│  - question     │──── generate ────►│  - method      ─┼─ composite key    │
│  - answer       │                   │  - lang        ─┤                   │
│  - extra        │                   │  - model       ─┘                   │
└─────────────────┘                   │  - gen:  {query}                    │
        │                             │  - exec: {result, success, error}   │
        │                             │  - eval: {exact_match, f1, ...}     │
        │                             └─────────────────────────────────────┘
        │                                       │
        └───────────── evaluate ────────────────┘
                   (compare answers)
```

### Database Schema

**src.db** (read-only) - Source records from dataset

| Table: `data` | Type | Description |
|---------------|------|-------------|
| `id` | TEXT | Primary key |
| `question` | TEXT | Natural language question |
| `answer` | TEXT | JSON array of ground truth answers |
| `extra` | TEXT | JSON object with optional fields (hop, split, ...) |

**dst.db** (read-write) - Experiment results

| Table: `data` | Type | Description |
|---------------|------|-------------|
| `question_id` | TEXT | ┐ |
| `method` | TEXT | │ Composite |
| `lang` | TEXT | │ Primary Key |
| `model` | TEXT | ┘ |
| `gen` | TEXT | JSON: {query} |
| `exec` | TEXT | JSON: {result, success, error} |
| `eval` | TEXT | JSON: {exact_match, f1, precision, recall} |

The composite key `(question_id, method, lang, model)` allows multiple experiment results for the same question.

## Progress

### LLM

| Dataset | Model | Lang | Schema | Data | Server | Gen | Exec | Eval | Acc | F1 |
|---------|-------|------|:------:|:----:|:------:|:---:|:----:|:----:|----:|---:|
| metaqa | deepseek-chat | cypher |  [x]   | [x] |  [x]   | [x] | [x] | [x] | 78.8% | 80.9% |
| metaqa | deepseek-chat | sparql |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| metaqa | deepseek-chat | gremlin |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| metaqa | deepseek-reasoner | cypher |  [x]   | [x] |  [x]   | [x] | [x] | [x] | 82.2% | 86.6% |
| metaqa | deepseek-reasoner | sparql |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| metaqa | deepseek-reasoner | gremlin |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-chat | cypher |  [ ]   | [ ] |  [x]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-chat | sparql |  [ ]   | [ ] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-chat | gremlin |  [ ]   | [ ] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-reasoner | cypher |  [ ]   | [ ] |  [x]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-reasoner | sparql |  [ ]   | [ ] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| openreview | deepseek-reasoner | gremlin |  [ ]   | [ ] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-chat | cypher |  [ ]   | [ ] | [N/A]  | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-chat | sparql |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-chat | gremlin |  [ ]   | [ ] | [N/A]  | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-reasoner | cypher |  [ ]   | [ ] | [N/A]  | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-reasoner | sparql |  [ ]   | [x] |  [ ]   | [ ] | [ ] | [ ] | - | - |
| webqsp | deepseek-reasoner | gremlin |  [ ]   | [ ] | [N/A]  | [ ] | [ ] | [ ] | - | - |

### GraphQ_IR

| Dataset | Lang |      Data Prep      |  Server Prep  | Pretrain | Posttrain | Gen | Exec | Eval | Acc | F1 |
|---------|------|:-------------------:|:-------------:|:--------:|:---------:|:---:|:----:|:----:|----:|---:|
| metaqa | cypher |         [x]         |      [x]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| metaqa | sparql |         [x]         |      [x]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| metaqa | gremlin |         [Challenging]          |      [ ]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| openreview | cypher |         [ ]         |      [x]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| openreview | sparql |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| openreview | gremlin |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| webqsp | cypher |    [Challenging]    | [Infeasible] | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| webqsp | sparql |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| webqsp | gremlin |    [Challenging]    | [Infeasible]  | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
