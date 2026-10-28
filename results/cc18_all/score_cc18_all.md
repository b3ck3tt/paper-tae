# CC18 re-run scoring — package A (72 tasks, 3073 catalog flows)

Family map: P4 §4.3. 'other' = 31.5 % of distinct candidate flows, 37.4 % of candidate rows.

## Complete / partial / missing

| model | complete ranking | partial ranking (median coverage) | missing |
|---|---|---|---|
| Haiku 4.5 | 11 | 61 (0.62) | 0 |
| Opus 4.8 | 17 | 55 (0.96) | 0 |
| Sonnet 5 + thinking | 16 | 55 (0.55) | 1 |
| Sonnet 5 | 7 | 65 (0.31) | 0 |
| Llama-3.3-70B-Turbo (FP8) | 17 | 55 (0.99) | 0 |
| Llama-3.1-8B-Turbo (FP8) | 23 | 49 (0.98) | 0 |
| GPT-5-mini | 58 | 14 (0.97) | 0 |
| GPT-5 | 42 | 30 (0.95) | 0 |

## Coverage and missing tasks

| model | tasks with ranking | missing after 3 attempts | truncated attempts / all attempts | median #cand (missing) | median #cand (ranked) |
|---|---|---|---|---|---|
| Haiku 4.5 | 72/72 | 0 | 0/72 | — | 399 |
| Opus 4.8 | 72/72 | 0 | 5/77 | — | 399 |
| Sonnet 5 + thinking | 71/72 | 1 (3903) | 16/87 | 509.0 | 398 |
| Sonnet 5 | 72/72 | 0 | 0/72 | — | 399 |
| Llama-3.3-70B-Turbo (FP8) | 72/72 | 0 | 39/102 | — | 399 |
| Llama-3.1-8B-Turbo (FP8) | 72/72 | 0 | 95/145 | — | 399 |
| GPT-5-mini | 72/72 | 0 | 0/72 | — | 399 |
| GPT-5 | 72/72 | 0 | 0/72 | — | 399 |

Common task set (every model has a ranking): **71** tasks.


## flow level vs flow portfolio (P4: best-on-average, task_group_kfold) — common set

| model | model nregret@1 | portfolio | Δ (model − portfolio) | 95 % CI | Wilcoxon p | BH q | n |
|---|---|---|---|---|---|---|---|
| Opus 4.8 | 0.080 | 0.090 | -0.010 | [-0.044, +0.025] | 0.44 | 0.71 | 71 |
| Sonnet 5 + thinking | 0.093 | 0.090 | +0.003 | [-0.031, +0.037] | 0.82 | 0.86 | 71 |
| Sonnet 5 | 0.101 | 0.090 | +0.011 | [-0.021, +0.044] | 0.85 | 0.86 | 71 |
| Haiku 4.5 | 0.109 | 0.090 | +0.020 | [-0.022, +0.066] | 0.86 | 0.86 | 71 |
| GPT-5 | 0.171 | 0.090 | +0.081 | [+0.028, +0.144] | 0.0067 | 0.013 | 71 |
| Llama-3.3-70B-Turbo (FP8) | 0.174 | 0.090 | +0.084 | [+0.034, +0.137] | 0.0014 | 0.0036 | 71 |
| Llama-3.1-8B-Turbo (FP8) | 0.370 | 0.090 | +0.280 | [+0.193, +0.369] | 3.7e-07 | 1.5e-06 | 71 |
| GPT-5-mini | 0.421 | 0.090 | +0.332 | [+0.238, +0.427] | 3.4e-08 | 2.7e-07 | 71 |
| **median over models** (per task) | 0.104 | 0.090 | +0.014 | [-0.018, +0.048] | 0.42 | — | 71 |

Models below the portfolio: 1 of 8 (BH q < 0.05: 0); above: 7 (BH q < 0.05: 4).

## flow level vs flow portfolio, sensitivity: leave-one-task-out within the package — common set

| model | model nregret@1 | portfolio | Δ (model − portfolio) | 95 % CI | Wilcoxon p | BH q | n |
|---|---|---|---|---|---|---|---|
| Opus 4.8 | 0.080 | 0.106 | -0.026 | [-0.069, +0.016] | 0.14 | 0.22 | 71 |
| Sonnet 5 + thinking | 0.093 | 0.106 | -0.013 | [-0.053, +0.026] | 0.52 | 0.69 | 71 |
| Sonnet 5 | 0.101 | 0.106 | -0.005 | [-0.043, +0.031] | 0.76 | 0.76 | 71 |
| Haiku 4.5 | 0.109 | 0.106 | +0.004 | [-0.044, +0.055] | 0.72 | 0.76 | 71 |
| GPT-5 | 0.171 | 0.106 | +0.066 | [+0.008, +0.131] | 0.057 | 0.11 | 71 |
| Llama-3.3-70B-Turbo (FP8) | 0.174 | 0.106 | +0.068 | [+0.020, +0.121] | 0.0091 | 0.024 | 71 |
| Llama-3.1-8B-Turbo (FP8) | 0.370 | 0.106 | +0.264 | [+0.181, +0.351] | 6.2e-07 | 2.5e-06 | 71 |
| GPT-5-mini | 0.421 | 0.106 | +0.316 | [+0.224, +0.410] | 7.4e-08 | 5.9e-07 | 71 |
| **median over models** (per task) | 0.104 | 0.106 | -0.001 | [-0.039, +0.035] | 0.83 | — | 71 |

Models below the portfolio: 3 of 8 (BH q < 0.05: 0); above: 5 (BH q < 0.05: 3).

## family level vs family portfolio (a) mean family accuracy — common set

| model | model nregret@1 | portfolio | Δ (model − portfolio) | 95 % CI | Wilcoxon p | BH q | n |
|---|---|---|---|---|---|---|---|
| GPT-5 | 0.104 | 0.135 | -0.031 | [-0.065, +0.003] | 0.034 | 0.09 | 71 |
| Sonnet 5 + thinking | 0.114 | 0.135 | -0.021 | [-0.045, -0.001] | 0.083 | 0.17 | 71 |
| Opus 4.8 | 0.122 | 0.135 | -0.012 | [-0.034, +0.011] | 0.026 | 0.09 | 71 |
| Haiku 4.5 | 0.132 | 0.135 | -0.003 | [-0.017, +0.015] | 0.44 | 0.51 | 71 |
| Sonnet 5 | 0.147 | 0.135 | +0.012 | [-0.019, +0.044] | 0.3 | 0.48 | 71 |
| GPT-5-mini | 0.168 | 0.135 | +0.033 | [-0.030, +0.097] | 0.94 | 0.94 | 71 |
| Llama-3.1-8B-Turbo (FP8) | 0.176 | 0.135 | +0.042 | [-0.009, +0.096] | 0.45 | 0.51 | 71 |
| Llama-3.3-70B-Turbo (FP8) | 0.212 | 0.135 | +0.078 | [+0.032, +0.129] | 0.0059 | 0.047 | 71 |
| **median over models** (per task) | 0.133 | 0.135 | -0.002 | [-0.022, +0.021] | 0.25 | — | 71 |

Models below the portfolio: 4 of 8 (BH q < 0.05: 0); above: 4 (BH q < 0.05: 1).

## family level vs family portfolio (b) mean best-in-family accuracy — common set

| model | model nregret@1 | portfolio | Δ (model − portfolio) | 95 % CI | Wilcoxon p | BH q | n |
|---|---|---|---|---|---|---|---|
| GPT-5 | 0.104 | 0.103 | +0.001 | [-0.032, +0.037] | 0.65 | 0.65 | 71 |
| Sonnet 5 + thinking | 0.114 | 0.103 | +0.011 | [-0.025, +0.047] | 0.62 | 0.65 | 71 |
| Opus 4.8 | 0.122 | 0.103 | +0.019 | [-0.022, +0.063] | 0.54 | 0.65 | 71 |
| Haiku 4.5 | 0.132 | 0.103 | +0.029 | [-0.011, +0.071] | 0.23 | 0.37 | 71 |
| Sonnet 5 | 0.147 | 0.103 | +0.044 | [-0.000, +0.090] | 0.06 | 0.16 | 71 |
| GPT-5-mini | 0.168 | 0.103 | +0.064 | [+0.007, +0.127] | 0.12 | 0.23 | 71 |
| Llama-3.1-8B-Turbo (FP8) | 0.176 | 0.103 | +0.073 | [+0.018, +0.138] | 0.017 | 0.069 | 71 |
| Llama-3.3-70B-Turbo (FP8) | 0.212 | 0.103 | +0.109 | [+0.055, +0.165] | 0.00024 | 0.0019 | 71 |
| **median over models** (per task) | 0.133 | 0.103 | +0.030 | [-0.007, +0.070] | 0.11 | — | 71 |

Models below the portfolio: 0 of 8 (BH q < 0.05: 0); above: 8 (BH q < 0.05: 1).

## Flow level, hit@1 / Spearman / coverage — common set

| model | hit@1 | Spearman | coverage |
|---|---|---|---|
| Haiku 4.5 | 0.00 | 0.14 | 0.69 |
| Opus 4.8 | 0.04 | 0.16 | 0.95 |
| Sonnet 5 + thinking | 0.03 | 0.22 | 0.68 |
| Sonnet 5 | 0.03 | 0.23 | 0.56 |
| Llama-3.3-70B-Turbo (FP8) | 0.00 | 0.02 | 0.92 |
| Llama-3.1-8B-Turbo (FP8) | 0.00 | 0.08 | 0.79 |
| GPT-5-mini | 0.00 | -0.01 | 0.98 |
| GPT-5 | 0.03 | 0.06 | 0.88 |

## All available tasks per model (NOT comparable across models)

| model | n | flow nregret@1 | family nregret@1 |
|---|---|---|---|
| Haiku 4.5 | 72 | 0.108 | 0.130 |
| Opus 4.8 | 72 | 0.079 | 0.121 |
| Sonnet 5 + thinking | 71 | 0.093 | 0.114 |
| Sonnet 5 | 72 | 0.099 | 0.145 |
| Llama-3.3-70B-Turbo (FP8) | 72 | 0.172 | 0.209 |
| Llama-3.1-8B-Turbo (FP8) | 72 | 0.365 | 0.181 |
| GPT-5-mini | 72 | 0.416 | 0.165 |
| GPT-5 | 72 | 0.169 | 0.103 |
