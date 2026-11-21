# Reproduce the paper splits and HyperHarness runs

Start from the `main` branch. The commands below do
not consume an old experiment, model answer, score, or selected candidate.
They register a fresh HyperHarness run on the published user and query splits
for Qwen3.5-9B, Qwen3.8-27B or GPT-5.6-Luna. The initial generator is checked
against the corresponding paper run for all nine combinations. This is not a
launcher for the separate GEPA, Meta-Harness or other baseline experiments.

| Benchmark | TRAIN users / queries | TEST users / queries |
|---|---:|---:|
| PersonaMemv3 | 25 / 1,860 | 75 / 5,700 |
| Mem-PAL (English) | 50 / 418 | 50 / 408 |
| MemoryCD (four tasks) | 100 / 1,200 | 223 / 2,676 |

## Install and fetch

On Linux, use Python 3.12 and a fresh virtual environment. Set `LOCAL` to a
node-local filesystem with enough space; the Amazon metadata build in particular
downloads all registered categories. Retain the source files and frozen run
registrations on durable storage. Do not copy active databases mid-write.

```bash
git clone -b main https://github.com/TamSiuhin/hyperharness-code.git
cd hyperharness-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[memorycd,eval,luna,dev]'
export LOCAL="$HOME/hyperharness-paper"
mkdir -p "$LOCAL/raw"
export PYTHONPATH="$PWD/src"
for dataset in pm3 mempal memorycd; do
  python scripts/paper_splits.py "$dataset" --fetch --source "$LOCAL/raw/$dataset" || exit 1
done
```

`--fetch` downloads only missing files and verifies SHA-256 against the tracked
locks in [configs/paper-splits](../configs/paper-splits). Omit it for offline
verification of an existing release. PM3 uses its original later-half query IDs
and timestamps, not a new random split. Mem-PAL regenerates the signed-selection
cases from the English release. MemoryCD uses the original user seed, three
held-out Home items and four tasks per item. These are scheduled items, not
post-filter counts for a particular model.

## Prepare native inputs

PM3 needs a clean node-local checkout of the official evaluator and its CPU
sentence embedder. The evaluator's full requirements include optional methods
that are not used in the HyperHarness study. MemoryCD needs the Amazon Reviews
2023 **title catalog** for negative candidates; the four-domain interaction
files alone do not contain all those titles. Build the public catalog once, or
use a copy of that catalog. Preparation rejects a different resulting candidate
set or user record even if the split counts agree.

```bash
PM3_EVALUATOR_DIR="$LOCAL/evaluator" bash scripts/fetch_pm3_evaluator.sh
python -m pip install -r "$LOCAL/evaluator/requirements.txt"
python scripts/prepare_horizon_task23.py metadata --public-categories \
  --cache "$LOCAL/amazon-cache" --catalog "$LOCAL/metadata/titles.parquet"
```

## Register and run

Registration does not call the answering model, optimizer, or judge. Each `--work`
must be new. It freezes the benchmark inputs and checks the initial generator's
SHA-256 against the paper run before printing the `study` command.

```bash
python scripts/paper_study.py pm3 --source "$LOCAL/raw/pm3" \
  --evaluator "$LOCAL/evaluator" --work "$LOCAL/runs/pm3"
python scripts/paper_study.py mempal --source "$LOCAL/raw/mempal" --work "$LOCAL/runs/mempal"
python scripts/paper_study.py memorycd --source "$LOCAL/raw/memorycd" \
  --catalog "$LOCAL/metadata/titles.parquet" --work "$LOCAL/runs/memorycd"
```

These examples default to `--executor 9b`. For the other six configurations,
repeat each registration with `--executor 27b` or `--executor luna` and a new
`--work` directory for each dataset/model pair. The run command always reads
the registered model. Luna keeps the same user roster. Its fixed Mem-PAL TRAIN
mask leaves 417/418 items; TEST retains all 408. MemoryCD keeps seven known
TRAIN exclusions and scores common unfiltered items without imputing zeros.
PM3 and MemoryCD use the targeted Luna TRAIN selection hooks.

To execute, provision a Qwen3.5-9B OpenAI-compatible server on
`http://127.0.0.1:8001/v1` (model ID `Qwen3.5-9B`, thinking disabled, structured
output support), or a `Qwen3.8-27B` server on port 8002 for 27B. MemoryCD's local
vLLM check requires the guidance backend with `disable_any_whitespace=true`.
Luna uses the hosted `gpt-5.6-luna` deployment instead of a local vLLM server.
Also provision the registered tokenizer, Codex CLI, the `gpt-6-sol` construction
and optimization deployment, and for PM3 the `gpt-5.5` judge. The existing
controllers require Linux network namespace isolation and `setpriv`; do not
disable the sandbox. Configure the provider credentials **outside this repo**.
Set `HYPERHARNESS_SOL_ENV` before registration to the absolute path of the Sol
credential file. The native Sol transport expects its registered Sweden Central
Azure route. The PM3 native executor/judge uses the Azure credential loader in
`scripts/run_lamp_skill.py` (Azure endpoint, API key and executor API key).
Set `HYPERHARNESS_EXECUTOR_CREDENTIALS` to its private file path; Luna reads the
same variable. `HF_HOME` can point at a node-local tokenizer cache. Never put
credential values in version-controlled configuration.
Shared caches and provider deployment access are separate from the public benchmark.

Run the printed commands, or equivalently:

```bash
for dataset in pm3 mempal memorycd; do
  python scripts/paper_study.py "$dataset" --run --work "$LOCAL/runs/$dataset" || exit 1
done
```

Each study evaluates r0 and optimizes on TRAIN for ten rounds. TEST registration
requires a complete selected TRAIN panel. Inspect `train/selection.json` and the
TRAIN and TEST `summary.json` files in the new run. These are fresh results, not
cached paper scores. Provider sampling, deployment revisions, and the public
direct transport (instead of the historical failover gateway) can change them.

### Optional Azure failover gateway

For a reference run using `--failover-credentials`, set
`HYPERHARNESS_AZURE_FAILOVER_HOSTS` in both the registration shell and the
gateway shell. List your three Azure OpenAI resource hostnames in failover
order, separated by commas. For example:

```bash
export HYPERHARNESS_AZURE_FAILOVER_HOSTS="primary-example.openai.azure.com,secondary-example.openai.azure.com,backup-example.openai.azure.com"
```

Pass the loopback client's credential file to `--failover-credentials`.
Pass the separate, owner-only upstream JSON file to
`scripts/azure_endpoint_failover.py --config`. Its `endpoints` array must use
the same order. The standard `--provider azure` workflow uses the single
`AZURE_OPENAI_ENDPOINT` from the README.
