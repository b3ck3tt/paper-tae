# Can We Trust "LLM-as-Recommender"? — code & data

Anonymous supplementary material for the TAE (NeurIPS 2026) submission
*"Can We Trust 'LLM-as-Recommender'? A Cross-Vendor, Difficulty-Graded Evaluation of Frozen LLMs for
AutoML Pipeline Selection."*

This repository lets a reviewer **reproduce every table from the released CSVs with no API keys and no
OpenML access**, and re-run the full evaluation from scratch if desired.

## Quick reproduction (no API, ~10 s)

```bash
pip install -r requirements.txt
python reproduce_tables.py
python family_level.py        # long-tail family-level cell of Table 4 + size of the "other" bucket
```

Prints Table 1 (per-model quality), Table 2 (mean-difference vs. portfolio, BH-corrected — 10/11 worse),
and Table 5 (task-invariance / degeneracy of top-1 picks), computed from `results/*.csv`.

## What's here

```
reproduce_tables.py          reproduce the main tables from the CSVs
family_level.py              family-level re-scoring of the long-tail rankings (Table 4, family x long tail)
requirements.txt
src/
  llm_paper4.py              evaluation harness (build | dryrun | run <model> [N] | eval)
  llm_sig.py                 paired significance, mixed-effects, BH correction, tier tests
results/
  ground_truth.csv           (task_id, candidate_id, accuracy) — the held-out labels, self-contained
  catalog.csv                (candidate_id, flow_id, flow_name) — candidate IDs mapped to OpenML flows
  rankings_main200.csv       per-model top-to-bottom rankings, 11 models x 200 tasks (+ token usage)
  llm_metrics_per_task.csv   per-task nregret@1 / hit@1 / Spearman, per model
  baselines_main200.csv      trained text-LTR / kNN / portfolio / random, per task (200-subset)
  rankings_leakage200_*.csv  leakage ablation (dataset name revealed), Sonnet & DeepSeek, n=200
  rankings_seed123pt_*.csv   independent second seed (stability), per-task catalog
```

## Re-running the LLM evaluation (needs API keys)

The harness infers the provider from the model id and reads each provider's key from an env var or
`~/.<provider>_key` (Anthropic / OpenAI / DeepInfra). The prompt (instructions + flow catalog) is cached;
open-weight models are served via an OpenAI-compatible endpoint.

```bash
cd src
python llm_paper4.py build                 # build task pack (needs the OpenML long-tail cache)
python llm_paper4.py run gpt-5             # or: run "deepinfra:deepseek-ai/DeepSeek-V3", run claude-sonnet-5, ...
python llm_paper4.py eval                  # score all runs vs the trained baselines
python llm_sig.py                          # significance tables
```

Notes: OpenAI reasoning models must run at low `reasoning_effort` (default effort does not reliably
terminate on large-candidate ranking prompts — see the paper's Threats section). DeepInfra does not
prefix-cache. Per-task token usage is logged (used for the effort analysis).

## License

Released for review. MIT.
