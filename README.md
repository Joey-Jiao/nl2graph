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

## Methods

The framework supports two generation methods: **LLM** (prompt-based) and **Seq2Seq** (fine-tuned BART).

### LLM Method

Prompt-based generation using commercial or open-source LLMs.

**Processing Pipeline:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM Generation                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │ Schema JSON │ ──── │ from_dict() │ ──── │ Schema      │                  │
│  │ (file)      │      │             │      │ Object      │                  │
│  └─────────────┘      └─────────────┘      └──────┬──────┘                  │
│                                                   │                         │
│                                          to_prompt_string()                 │
│                                                   │                         │
│                                                   ▼                         │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │ Question    │ ──── │ Jinja2      │ ◄─── │ Schema      │                  │
│  │ (text)      │      │ Template    │      │ String      │                  │
│  └─────────────┘      └──────┬──────┘      └─────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│                       ┌─────────────┐                                       │
│                       │ Prompt      │                                       │
│                       │ (text)      │                                       │
│                       └──────┬──────┘                                       │
│                              │                                              │
│                         LLM API                                             │
│                              │                                              │
│                              ▼                                              │
│                       ┌─────────────┐      ┌─────────────┐                  │
│                       │ Raw Output  │ ──── │ Extract     │                  │
│                       │ (text)      │      │ Query       │                  │
│                       └─────────────┘      └──────┬──────┘                  │
│                                                   │                         │
│                                                   ▼                         │
│                                            ┌─────────────┐                  │
│                                            │ Query       │                  │
│                                            │ (cypher/    │                  │
│                                            │  sparql/    │                  │
│                                            │  gremlin)   │                  │
│                                            └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Input:**
- `question`: Natural language question
- `schema`: Graph schema JSON file (converted to structured text)

**Output:**
- `query`: Generated graph query
- `stats`: Token usage and latency

**Schema Processing:**

Schema JSON is NOT passed directly to LLM. It undergoes transformation:

```
schema.json                    CypherSchema.to_prompt_string()
─────────────────────────────  ─────────────────────────────────────
{                              Graph: MovieDB
  "name": "MovieDB",
  "nodes": [                   Nodes:
    {                            (Movie) [title: string]
      "label": "Movie",          (Person) [name: string]
      "properties": {
        "title": "string"      Edges:
      }                          (:Person)-[:ACTED_IN]->(:Movie)
    },                           (:Person)-[:DIRECTED]->(:Movie)
    {
      "label": "Person",
      "properties": {
        "name": "string"
      }
    }
  ],
  "edges": [
    {
      "label": "ACTED_IN",
      "source_label": "Person",
      "target_label": "Movie"
    },
    {
      "label": "DIRECTED",
      "source_label": "Person",
      "target_label": "Movie"
    }
  ]
}
```

**SPARQL Schema Example:**

```
schema.json                              SparqlSchema.to_prompt_string()
───────────────────────────────────────  ─────────────────────────────────────
{                                        RDF Graph: MovieDB
  "name": "MovieDB",
  "prefixes": {                          Prefixes:
    "": "http://example.org/",             : <http://example.org/>
    "rdfs": "http://...rdf-schema#"        rdfs: <http://...rdf-schema#>
  },
  "classes": [                           Classes:
    {"uri": ":Person"},                    :Movie
    {"uri": ":Movie"}                      :Person
  ],
  "properties": [                        Properties:
    {                                      :actedIn domain=:Person range=:Movie [ObjectProperty]
      "uri": ":actedIn",                   :directed domain=:Person range=:Movie [ObjectProperty]
      "domain": ":Person",                 :name domain=:Person range=xsd:string [DatatypeProperty]
      "range": ":Movie",                   :title domain=:Movie range=xsd:string [DatatypeProperty]
      "is_object_property": true
    },
    ...
  ]
}
```

**Gremlin Schema Example:**

```
schema.json                    GremlinSchema.to_prompt_string()
─────────────────────────────  ─────────────────────────────────────
{                              Graph: MovieDB
  "name": "MovieDB",
  "nodes": [                   Vertices:
    {                            Movie [title: string]
      "label": "Movie",          Person [name: string]
      "properties": {...}
    },                         Edges:
    ...                          Person -[ACTED_IN]-> Movie
  ],                             Person -[DIRECTED]-> Movie
  "edges": [
    {
      "label": "ACTED_IN",
      "source_label": "Person",
      "target_label": "Movie"
    },
    ...
  ]
}
```

### Seq2Seq Method

Fine-tuned BART model for direct question-to-query translation.

**Processing Pipeline:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Seq2Seq Generation                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │ Question    │ ──── │ Tokenizer   │ ──── │ Input IDs   │                  │
│  │ (text)      │      │             │      │ (tensor)    │                  │
│  └─────────────┘      └─────────────┘      └──────┬──────┘                  │
│                                                   │                         │
│                                              BART Model                     │
│                                                   │                         │
│                                                   ▼                         │
│                                            ┌─────────────┐                  │
│                                            │ Output IDs  │                  │
│                                            │ (tensor)    │                  │
│                                            └──────┬──────┘                  │
│                                                   │                         │
│                                              Tokenizer                      │
│                                               Decode                        │
│                                                   │                         │
│                                                   ▼                         │
│                       ┌───────────────────────────────────────────┐         │
│                       │                                           │         │
│                       ▼                                           ▼         │
│  [IR Mode OFF]  ┌─────────────┐          [IR Mode ON]  ┌─────────────┐      │
│                 │ Query       │                        │ IR          │      │
│                 │ (cypher/    │                        │ (graphq-ir) │      │
│                 │  sparql)    │                        └──────┬──────┘      │
│                 └─────────────┘                               │             │
│                                                          Translator         │
│                                                               │             │
│                                                               ▼             │
│                                                        ┌─────────────┐      │
│                                                        │ Query       │      │
│                                                        │ (cypher/    │      │
│                                                        │  sparql/    │      │
│                                                        │  kopl)      │      │
│                                                        └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Input:**
- `question`: Natural language question
- `schema`: NOT used (schema knowledge is embedded during training)

**Output:**
- `query`: Generated graph query (or IR if `--ir` mode)

**IR Mode:**

When `--ir` flag is enabled, the model outputs an intermediate representation (GraphQ-IR) which is then translated to the target query language:

```
Question ──► BART ──► GraphQ-IR ──► Translator ──► Query
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                    to_cypher()    to_sparql()     to_kopl()
```

### Method Comparison

| Aspect | LLM | Seq2Seq |
|--------|-----|---------|
| Schema | Required (runtime) | Not used (training-time) |
| Model | API-based (GPT, DeepSeek) | Local (fine-tuned BART) |
| Training | None | Required |
| Latency | Higher (API call) | Lower (local inference) |
| Cost | Per-token billing | One-time training |
| Flexibility | High (prompt engineering) | Low (retraining needed) |

## Data Flow

```
src.db (read-only)                    dst.db (read-write)
┌─────────────────┐                   ┌─────────────────────────────────────┐
│ Record          │                   │ Result                              │
│  - id           │                   │  - question_id ─┐                   │
│  - question     │──── generate ────►│  - method      ─┼─ composite key    │
│  - answer       │                   │  - lang        ─┤                   │
│  - extra        │                   │  - model       ─┘                   │
└─────────────────┘                   │  - gen:  {query, stats}             │
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

| Table: `data` | Type | Description                                |
|---------------|------|--------------------------------------------|
| `question_id` | TEXT | ┐                                          |
| `method` | TEXT | │ Composite                                |
| `lang` | TEXT | │ Primary Key                              |
| `model` | TEXT | ┘                                          |
| `gen` | TEXT | JSON: {query, stats}                       |
| `exec` | TEXT | JSON: {result, success, error}             |
| `eval` | TEXT | JSON: {exact_match, f1, precision, recall} |

The composite key `(question_id, method, lang, model)` allows multiple experiment results for the same question.

## Progress

### LLM

| Dataset | Model | Lang | Schema | Data | Server | Gen | Exec | Eval | Acc | F1 |
|---------|-------|------|:------:|:----:|:------:|:---:|:----:|:----:|----:|---:|
| metaqa | deepseek-chat | cypher |  [x]   | [x]  |  [x]   | [x] | [x]  | [x]  | 80.6% | 82.8% |
| metaqa | deepseek-chat | sparql |  [x]   | [x]  |  [x]   | [x] | [ ]  | [ ]  | - | - |
| metaqa | deepseek-chat | gremlin |  [x]   | [x]  |  [x]   | [x] | [x]  | [x]  | 53.6% | 64.7% |
| metaqa | deepseek-reasoner | cypher |  [x]   | [x]  |  [x]   | [x] | [x]  | [x]  | 85.0% | 90.9% |
| metaqa | deepseek-reasoner | sparql |  [x]   | [x]  |  [x]   | [x] | [ ]  | [ ]  | - | - |
| metaqa | deepseek-reasoner | gremlin |  [x]   | [x]  |  [x]   | [x] | [ ]  | [ ]  | - | - |
| openreview | deepseek-chat | cypher |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |
| openreview | deepseek-chat | sparql |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |
| openreview | deepseek-chat | gremlin |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |
| openreview | deepseek-reasoner | cypher |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |
| openreview | deepseek-reasoner | sparql |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |
| openreview | deepseek-reasoner | gremlin |  [x]   | [ ]  |  [x]   | [ ] | [ ]  | [ ]  | - | - |

### GraphQ_IR

| Dataset | Lang |      Data Prep      |  Server Prep  | Pretrain | Posttrain | Gen | Exec | Eval | Acc | F1 |
|---------|------|:-------------------:|:-------------:|:--------:|:---------:|:---:|:----:|:----:|----:|---:|
| metaqa | cypher |         [x]         |      [x]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| metaqa | sparql |         [x]         |      [x]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| metaqa | gremlin |         [Challenging]          |      [ ]      | [x] | [x] | [ ] | [ ] | [ ] | - | - |
| openreview | cypher |         [ ]         |      [x]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| openreview | sparql |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| openreview | gremlin |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
| webqsp | sparql |         [ ]         |      [ ]      | [x] | [ ] | [ ] | [ ] | [ ] | - | - |
