# PLAGUE: Plug-and-play Framework for Lifelong Adaptive Generation of Multi-turn Jailbreaks

Reference implementation for the paper:

> **PLAGUE: Plug-and-play Framework for Lifelong Adaptive Generation of Multi-turn Jailbreaks**
> Neeladri Bhuiya*, Madhav Aggarwal*, Diptanshu Purwar — *A10 Networks, Inc.*
> *ICLR 2026 (Poster).*
> Paper PDF: [`plague_main.pdf`](./plague_main.pdf) &nbsp;|&nbsp; OpenReview: https://openreview.net/forum?id=05hNleYOcG &nbsp;|&nbsp; PDF (online): https://openreview.net/pdf?id=05hNleYOcG

> ⚠️ **Responsible-use notice.** This repository is a red-teaming / AI-safety research tool for measuring the multi-turn robustness of frontier LLMs. It is intended for authorized safety evaluation only. Do not use it to cause real-world harm or to violate any provider's terms of service.

---

<p align="center">
  <a href="./plague_main.pdf">
    <img src="./docs/paper_preview.png" alt="PLAGUE framework overview: Primer, Planner and Finisher modules with a lifelong strategy library" width="100%">
  </a>
  <br>
  <em>The PLAGUE framework — Primer, Planner and Finisher phases with a lifelong strategy library. Click the figure to open the <a href="./plague_main.pdf">full paper</a>.</em>
</p>

## Overview

PLAGUE is a plug-and-play framework for building **multi-turn** jailbreak attacks, inspired by lifelong-learning agents. It dissects the lifetime of a multi-turn attack into three carefully designed phases:

- **Primer** – initializes the conversation context and warms up the target toward the harmful goal.
- **Planner** – plans and adapts the multi-turn trajectory, optionally retrieving from a growing **strategy library** (the "lifelong learning" component).
- **Finisher** – drives the conversation to completion (a GOAT- or Crescendo-style finisher) and elicits the final harmful response.

Across leading models PLAGUE improves attack success rate (ASR) by more than 30% within a comparable or smaller query budget. Using the StrongReject rubric, the paper reports an ASR of **81.4% on OpenAI o3** and **67.3% on Claude Opus 4.1** — two models considered highly resistant to jailbreaks.

## Repository layout

```
plague/
├── README.md
├── plague_main.pdf            # the published paper (ICLR 2026)
├── pyproject.toml              # installable `plague` package
├── requirements.txt
├── .env.example               # API keys template
├── data/
│   └── harmbench_dataset/      # HarmBench prompts (HuggingFace `datasets` format)
├── scripts/
│   └── run_plague.py           # run an attack against one target model
└── src/plague/
    ├── plague.py               # `Plague` orchestrator (Primer/Planner/Finisher)
    ├── config.py               # `PlagueConfig`
    ├── blackbox_model.py       # unified model interface (OpenAI/Anthropic/Together/Gemini/vLLM)
    ├── infer.py                # provider query functions
    ├── evaluator.py            # ASR / StrongReject / JailbreakBench evaluators
    ├── rubric_based_scorer.py  # rubric scorer used during optimization
    ├── prompt_utils.py         # attack/eval prompts and helpers
    ├── prompts.py              # EVAL_PROMPT / TAP_PROMPT judge prompts
    ├── strategy_utils.py       # strategy-library retrieval
    ├── strategy_generator/     # adversarial strategy base classes
    └── candidate_generator/    # GOAT / Crescendo finishers
```

## Installation

Requires Python ≥ 3.11.

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

## Configuration (API keys)

PLAGUE talks to model providers through `python-dotenv`. Copy the template and fill in the keys you need:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY / ANTHROPIC_API_KEY / TOGETHER_API_KEY / GEMINI_API_KEY
```

- **Target models** (`o3`, `o1`, `gpt-4o`, `claude-opus-4-1`, DeepSeek-R1, …) are reached via the OpenAI / Anthropic / Together / Gemini APIs.
- The **attacker, evaluator, summariser and embedding** models in the paper are open-weight (DeepSeek-R1, Qwen3-235B, Llama-3.1-8B, Qwen3-Embedding-0.6B) and can be served either through Together AI or locally with vLLM.

### Optional: local model serving (vLLM)

When serving the open-weight helper models locally, PLAGUE expects them on these ports (OpenAI-compatible endpoints at `http://localhost:<port>/v1`):

| Role | Default model | Port |
|------|---------------|------|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | `7005` |
| Evaluator / rubric scorer | `Qwen/Qwen3-235B-A22B-fp8-tput` | `8001` |
| Summariser | `meta-llama/Llama-3.1-8B-Instruct` | `7000` |
| Local attacker / GOAT finisher | attacker model | `8201` |

Example launch:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 7005
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 7000
```

## Running your own evaluations

Use `scripts/run_plague.py` to run PLAGUE against any target model and sweep the
configuration however you like. Each flag maps onto a field of `PlagueConfig`,
so you can vary the target/attacker/evaluator models, the number of rounds, the
primer steps, the finisher type, and the planner / strategy-library toggles.

```bash
# Attack o3 with the default (canonical) configuration on 200 HarmBench goals.
python scripts/run_plague.py --target-model o3-2025-04-16 --num-goals 200

# Attack Claude Opus 4.1 with a GOAT finisher and the strategy library enabled.
python scripts/run_plague.py \
    --target-model claude-opus-4-1-20250805 \
    --finisher-type goat \
    --use-strategy-library

# Quick smoke test on a handful of goals.
python scripts/run_plague.py --target-model gpt-4o --num-goals 5
```

See `python scripts/run_plague.py --help` for the full set of flags.

Each run writes a self-contained directory under `results/<project>/plague_<N>/`:

- `config.json` – the exact `PlagueConfig` used.
- `plague_evals.csv` – per-goal queries, responses and judgments.
- `overall_results.json` – ASR per iteration under three judges: HarmBench-style
  (`ASR at <i>`), **StrongReject** (`StrongReject at <i>`) and JailbreakBench
  (`jailbreak_bench_evaluator at <i>`).
- `strategy_library.json` – the learned multi-turn strategies.

## Headline results (StrongReject ASR)

| Target model | ASR |
|--------------|-----|
| OpenAI o3 (`o3-2025-04-16`) | **81.4%** |
| Claude Opus 4.1 (`claude-opus-4-1-20250805`) | **67.3%** |

PLAGUE improves ASR by **>30%** over prior multi-turn attacks at a comparable or
smaller query budget. See the paper for the full results across all evaluated
models and judges.

## Data & acknowledgements

Evaluation uses the **HarmBench** "standard" behaviors (200 prompts), bundled
under `data/harmbench_dataset/` and redistributed under the MIT License — see
[`data/harmbench_dataset/LICENSE`](./data/harmbench_dataset/LICENSE).

## Citation

```bibtex
@inproceedings{bhuiya2026plague,
  title     = {PLAGUE: Plug-and-play Framework for Lifelong Adaptive Generation of Multi-turn Jailbreaks},
  author    = {Bhuiya, Neeladri* and Aggarwal, Madhav* and Purwar, Diptanshu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=05hNleYOcG}
}
```

## Attribution

Authors: **Neeladri Bhuiya\*, Madhav Aggarwal\*, Diptanshu Purwar** — *A10 Networks, Inc.*

Developed at and © 2026 **A10 Networks, Inc.** All rights reserved.
This research is released for AI-safety and red-teaming purposes.
