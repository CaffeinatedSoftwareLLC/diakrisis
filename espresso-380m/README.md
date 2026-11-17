---
license: apache-2.0
base_model: vllm-sr/vela3-fast
library_name: peft  # TODO: confirm once the inference wrapper exists
pipeline_tag: zero-shot-classification  # TODO: confirm; closest Hub tag for typed decisions
language: en
tags:
  - typed-decisions
  - classification
  - prompt-injection-detection
  - distillation
  - lora
  - encoder
---

![Diakrisis espresso 380m](banner-espresso-light.svg)
<!-- Upload banner-espresso-light.svg to the model repo root so this relative path resolves. -->

# diakrisis/espresso-380m

A 380M-parameter encoder that answers typed questions about a text and returns a probability for each answer. It is distilled from Cloudflare Clef-flash (9.4B) and scores about the same as that teacher overall on 19 audit suites (0.795 vs 0.793).

Diakrisis is Greek for "discernment, telling things apart". Espresso is the first model in the family.

## What it does

You give it a text (the "state") and one or more typed questions. It returns a probability for each possible answer.

| Question type | What it answers |
|---|---|
| `noul` | yes or no |
| `choice` | one option from a list, which can include "none of these" |
| `score` | one level on an ordered scale |

The request format is the same one used by Clef and Jev.

<!-- TODO: confirm the option limit. The brief says up to 255 options; the synth_options_xl suite has more than 255. -->

What is inside:

- Base encoder: `vllm-sr/vela3-fast` (itself based on `dmis-lab/UltraBERT-380M`).
- A LoRA adapter (rank 16) plus small decision heads.
- A frozen decision rule, `R12_DECISION_RULE.json`.

## Quick start

```python
# TODO: placeholder. The inference wrapper is still being written.
# This block will show: load the model, send a state plus typed questions,
# read back probabilities and any REVIEW result.
```

## Decision rule

The rule is frozen and ships with the model. It applies only to `choice` questions with at least 50 options that include "none of these":

1. Add a boost of 0.25 to "none of these".
2. Add a penalty of −0.15 to PASS.
3. If the top answer's confidence is below 0.88, return **REVIEW** instead of an answer.

The aim is to send uncertain many-option cases to a human instead of answering wrong without warning. The targets were: at most 25% sent to review, under 5% silent misses, under 5% silent wrong answers.

## Evaluation results

**Confirmation** (run once, on fresh items never used for tuning): passed.

| Check | Result |
|---|---|
| Gates on the main confirmation set | 17 of 17 passed |
| Many-option set, sent to review | 23.2% (209/900) |
| Many-option set, silent misses | 2.3% (7/300) |
| Many-option set, silent wrong answers | 2.2% (13/600) |
| Long documents, change against the base model | 0.0008 absolute, agreement 0.963 |

**Gains over the base model** (vela3-fast) on the 19 audit suites:

| Measure | Base | Espresso |
|---|---|---|
| Overall accuracy | 0.719 | 0.795 |
| Prompt-injection AUROC | 0.843 | 0.942 |

## Evaluation status (honest)

- **The audit did not fully pass.** It has 9 retention checks against the base model. Espresso passed 7. The two misses are narrow:
  - Long-input retention: the lower 95% bound was −0.0200; the limit was above −0.02.
  - Policy originals: −1.1 points; the limit was −1.0 point.
- v1.0 is released anyway, with this result stated here.
- **The 19 audit suites are not fresh data.** They are held-out test splits, but they were used for evaluation in earlier development rounds. Only the confirmation set above was untouched.
- **A hosted model scores higher.** Jev 1.14 (TypeSafe) scores 0.821 overall on the same suites.
- In the comparison below, a REVIEW from Espresso counts as a wrong answer.

## Comparison with other models

Overall accuracy on the 19 audit suites: same items, same scorer.

| Model | Overall | Overall without synth_options_xl |
|---|---|---|
| Jev 1.14 (TypeSafe, hosted API) | 0.821 | 0.824 |
| **espresso-380m** | **0.795** | **0.794** |
| Clef-flash (teacher, 9.4B) | 0.793 | 0.794 |
| gliclass-v4-edge | 0.745 | 0.749 |
| vela3-fast (base) | 0.719 | 0.720 |
| open-jev-deberta | 0.600 | 0.603 |
| laya | 0.519 | 0.522 |

Notes:

- Gemini Decision 1 (Google) has no complete predictions on these suites, so it has no row.
- The hosted APIs reject choice questions with more than 255 options. Those items count as wrong in the first column. The second column leaves that suite out for every model.
- Jev, jevbench and tasksource_jev all use TypeSafe's request format, so Jev may have a home advantage on those two suites.

Where Espresso leads its teacher:

| Suite | Espresso | Clef-flash |
|---|---|---|
| Prompt-injection detection (pi_detect) | 0.848 | 0.620 |
| Very many options (synth_options_xl) | 0.957 | 0.630 |
| Customer-service decisions (abcd_test) | 0.924 | 0.898 |

Where Espresso trails its teacher:

| Suite | Espresso | Clef-flash |
|---|---|---|
| Multi-step dependent questions (synth_multi_dependent) | 0.424 | 0.591 |
| typed_synth | 0.738 | 0.865 |
| jevbench | 0.664 | 0.757 |

<!-- TODO: link the full per-suite table (COMPARISON.md) once it has a public URL. -->

## Training data and licences

Training targets are the output distributions of the Clef-flash teacher (Apache-2.0).

| Source | Licence |
|---|---|
| Generated by this project (options, steer, synth, typed-synth parts) | Caffeinated Software LLC |
| ABCD (`asappresearch/abcd`) | MIT |
| LLMail-Inject (`microsoft/llmail-inject-challenge`) | MIT |
| PromptShield (`hendzh/PromptShield`) | Apache-2.0 |
| DynaBench | MIT |
| typed-decisions-synth | MIT |
| tasksource typed decisions | commercial-licensed rows only |
| LEDGAR and SCOTUS (LexGLUE) | CC-BY-4.0 |
| CLINC150 | CC-BY-3.0 |

Not in the training data:

- No outputs from proprietary models. Jev-labelled data was quarantined.
- No SQuAD (share-alike, dropped).
- No tasksource rows without a commercial licence.

Hardware: AWS g6e instances (one NVIDIA L40S). Total project compute was about $200.

## Intended use and limits

Intended use: fast typed decisions over text, such as routing, policy checks, option selection and prompt-injection screening.

Limits:

- English only.
- Not for medical, legal or safety-critical decisions without human review.
- A REVIEW output needs a human. Do not treat it as an answer.
- Inputs above about 23k tokens are untested.
- Weak on multi-step dependent questions (0.424).

## Attribution / NOTICE

```
diakrisis/espresso-380m - Copyright 2026 Caffeinated Software LLC. Licensed under the Apache License, Version 2.0.

Fine-tuned from vllm-sr/vela3-fast (Apache-2.0), itself based on dmis-lab/UltraBERT-380M (Apache-2.0).
Distilled from Cloudflare Clef-flash (Apache-2.0).
Training data includes: LEDGAR and SCOTUS from LexGLUE (CC-BY-4.0); CLINC150 (CC-BY-3.0);
DynaBench (MIT); typed-decisions-synth (MIT);
tasksource typed decisions (commercial-licensed rows only);
ABCD (MIT); LLMail-Inject (MIT); PromptShield (Apache-2.0).
```

## Citation

```bibtex
@misc{diakrisis_espresso_380m_2026,
  title  = {diakrisis/espresso-380m: a 380M typed-decision model},
  author = {{Caffeinated Software LLC}},
  year   = {2026},
  url    = {https://huggingface.co/diakrisis/espresso-380m}
}
```
