# Frozen LLMs as Recommenders for AutoML Pipeline Selection: Can We Trust the Evaluation? — code & data

Code and data for the paper

> Roman Neruda, Juan Carlos Figueroa-García, Carlos Alberto Franco Franco.
> *Frozen LLMs as Recommenders for AutoML Pipeline Selection: Can We Trust the Evaluation?*
> TAE (Trust-AI-Eval): Can We Trust AI Evaluation? — workshop at NeurIPS 2026.

This repository lets anyone **reproduce every table from the released CSVs with no API keys and no
OpenML access**, and re-run the full evaluation from scratch if desired.

## Quick reproduction (no API, ~10 s)

```bash
pip install -r requirements.txt
python reproduce_tables.py
python table4.py              # Table 4 (CC18 and long tail, flow and family level), Turbo control, Appendix A
```

`reproduce_tables.py` prints Table 1 (per-model quality), Table 2 (mean-difference vs. portfolio,
BH-corrected — 10/11 worse) and Table 5 (task-invariance / degeneracy of top-1 picks). `table4.py` prints
Table 4 with and without the "other" family bucket, the Llama-Turbo (FP8) control and the DeepSeek-V3
serving-determinism check of Appendix A. Both read only `results/`.

## What's here

```
reproduce_tables.py          reproduce Tables 1, 2, 5 from the CSVs
table4.py                    reproduce Table 4 + Turbo control + Appendix A determinism check
requirements.txt
src/
  llm_paper4.py              evaluation harness (build | dryrun | run <model> [N] | eval); per-call API log,
                             up to 3 attempts for empty/truncated answers, spend guard
  llm_sig.py                 paired significance, mixed-effects, BH correction, tier tests
  llm_pack_cc18_p4.py        CC18 task packs for the harness (same instructions / meta-features / format)
  llm_pack_turbo50.py        50-task long-tail pack for the Llama-Turbo control
  run_cc18_rerun.py          resumable driver of the October 2026 CC18 re-run
  score_cc18_rerun.py        scoring used for the re-run (reads the authors' internal directory layout;
                             table4.py is the self-contained equivalent on the released CSVs)
results/
  ground_truth.csv           (task_id, candidate_id, accuracy) — the held-out labels, self-contained
  catalog.csv                (candidate_id, flow_id, flow_name) — candidate IDs mapped to OpenML flows
  rankings_main200.csv       per-model top-to-bottom rankings, 11 models x 200 tasks (+ token usage)
  llm_metrics_per_task.csv   per-task nregret@1 / hit@1 / Spearman, per model
  baselines_main200.csv      trained text-LTR / kNN / portfolio / random, per task (200-subset)
  rankings_leakage200_*.csv  leakage ablation (dataset name revealed), Sonnet & DeepSeek, n=200
  rankings_seed123pt_*.csv   independent second seed (stability), per-task catalog
  cc18/                      October 2026 CC18 re-run (69 tasks with >= 15 scikit-learn candidates)
    ground_truth_cc18.csv    (task_id, candidate_id, flow_id, flow_name, accuracy)
    rankings_cc18.csv        11 models x 69 tasks: ranking, coverage, truncation, chosen attempt, tokens
    portfolio_cc18.csv       per-task nregret@1 of the flow portfolio (trained without the evaluated task)
    api_log_cc18.jsonl       one line per API call: request/response id, served model, stop reason,
                             token usage, timestamp, number of parsed ids
    reask_archive_cc18.jsonl earlier attempts of the tasks that were re-asked (see below)
    rankings_cc18_deepseek_aug2026.csv  the August 2026 DeepSeek-V3 run (no API log), for Appendix A
  turbo50/                   Llama-Turbo (FP8) control on 50 long-tail tasks: rankings, task list, API log
```

## CC18 re-run (October 2026) and changes to Table 4

The CC18 column of Table 4 comes from a re-run on 27 Oct 2026 of all eleven configurations with this
harness and prompt on the 69 OpenML-CC18 tasks with at least 15 scikit-learn candidates (catalog 1,475
flows). Every call is logged in `results/cc18/api_log_cc18.jsonl`. Details:

- Llama-3.3-70B and Llama-3.1-8B are no longer served under their original ids; the re-run uses the FP8
  "-Turbo" variants. A 50-task long-tail control (`results/turbo50/`) compares them with the original runs:
  70B matches (paired difference -0.006, 95 % CI [-0.048, +0.031]), 8B is somewhat worse (+0.066,
  [-0.009, +0.140]).
- Each task is asked up to three times if the answer is empty or truncated. A normally finished answer with
  a ranking is used; otherwise the attempt with the longest valid ranking (selection by length only). For
  nine tasks whose longer partial ranking had been lost by an earlier version of the harness, the task was
  asked again; the earlier attempts are kept in `reask_archive_cc18.jsonl`. (The 200-task main study used one
  attempt per task.)
- Portfolios are computed without the evaluated task: the flow portfolio of the main study (task-grouped CV)
  and two leave-one-task-out family portfolios, (a) by mean member accuracy and (b) by best member.

Changes relative to the submitted version of the paper:

- **Flow / CC18 cell.** The submitted value (DeepSeek-V3 0.141 vs. portfolio 0.070) used a portfolio that
  averaged flow accuracies over all CC18 tasks *including the evaluated task*. With the leakage-free
  portfolio of the main study the portfolio scores 0.090. The DeepSeek-V3 rankings also changed between the
  August run and the logged October re-run (identical prompt, temperature 0; top-1 differs on 24 of 69 tasks);
  Table 4 uses the logged run (0.086). Both runs are released.
- **Family / CC18 cell.** The submitted value (0.060 vs. 0.095) did not come from an LLM API call: the
  per-task family scores were written by the AI coding assistant used in the project, acting as the "frozen
  LLM" on the anonymized meta-features. The files are kept, marked as quarantined, in the authors' working
  repository and are not used anywhere; the cell is replaced by the API re-run above.
- **Family / long-tail cell.** The earlier `family_level.py` ordered the family portfolio by mean accuracy
  over all 200 tasks, including the evaluated one. `table4.py` uses leave-one-task-out portfolios (a) and (b)
  on the common 161-task set.
- **Family baseline.** With portfolio (a) the LLMs appear to win at the family level; with portfolio (b) no
  model wins on either benchmark. Results without the "other" family bucket are printed as well.


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

## Citation

```bibtex
@inproceedings{neruda2026frozenllm,
  author    = {Neruda, Roman and Figueroa-Garc{\'i}a, Juan Carlos and Franco Franco, Carlos Alberto},
  title     = {Frozen {LLMs} as Recommenders for {AutoML} Pipeline Selection: Can We Trust the Evaluation?},
  booktitle = {TAE (Trust-AI-Eval): Can We Trust AI Evaluation? Workshop at NeurIPS 2026},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`.
