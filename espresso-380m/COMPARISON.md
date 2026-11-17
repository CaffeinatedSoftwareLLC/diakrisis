# Model comparison on the 19 audit suites (same items, same scorer)

Accuracy per suite (questions in brackets). The candidate uses its frozen decision rule; REVIEW counts as not correct. n/a = the model has no complete predictions for that suite. Devices differ (aws-l40s, cuda, gpyou); scores are decisions, not latency.

| Suite | espresso-380m | vela3-fast | espresso R2 candidate | clef-flash | clef-text-0.6b | gliclass-v4-edge | gliner2.5-decide | laya | open-jev-deberta | Jev 1.14 | Gemini Decision 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abcd_injection (12600) | 0.891 | 0.713 | 0.885 | 0.840 | 0.763 | 0.776 | 0.609 | 0.464 | 0.604 | 0.855 | 0.884 |
| abcd_test (7000) | 0.924 | 0.850 | 0.923 | 0.898 | 0.786 | 0.850 | 0.714 | 0.579 | 0.659 | 0.902 | 0.898 |
| abstain (2000) | 0.838 | 0.830 | 0.811 | 0.854 | 0.608 | 0.735 | 0.472 | 0.539 | 0.596 | 0.895 | 0.933 |
| fast_decisions (2600) | 0.557 | 0.559 | 0.553 | 0.615 | 0.577 | 0.632 | 0.679 | 0.530 | 0.581 | 0.629 | 0.639 |
| injection_public (600) | 0.965 | 0.963 | 0.970 | 1.000 | 0.978 | 0.987 | 0.940 | 0.870 | 0.967 | 0.993 | 1.000 |
| jevbench (1500) | 0.664 | 0.667 | 0.661 | 0.757 | 0.571 | 0.723 | 0.452 | 0.538 | 0.404 | 0.725 | 0.732 |
| kotoba_ood (8000) | 0.639 | 0.628 | 0.638 | 0.665 | 0.528 | 0.692 | 0.584 | 0.462 | 0.689 | 0.735 | 0.716 |
| long_inputs (800) | 0.744 | 0.751 | 0.748 | 0.760 | n/a | 0.751 | n/a | 0.312 | 0.395 | 0.740 | 0.724 |
| many_options (2000) | 0.934 | 0.933 | 0.935 | 0.928 | n/a | 0.923 | 0.744 | 0.635 | 0.712 | 0.901 | 0.909 |
| perturb (5850) | 0.860 | 0.810 | 0.862 | 0.906 | n/a | 0.828 | 0.710 | 0.603 | 0.688 | 0.903 | 0.903 |
| pi_detect (3000) | 0.848 | 0.657 | 0.853 | 0.620 | n/a | 0.592 | 0.489 | 0.714 | 0.476 | 0.702 | 0.730 |
| policy (698) | 0.613 | 0.612 | 0.653 | 0.638 | n/a | 0.625 | 0.622 | 0.609 | 0.155 | 0.872 | 0.858 |
| synth_injection_typed (300) | 0.567 | 0.520 | 0.540 | 0.600 | n/a | 0.540 | 0.123 | 0.267 | 0.290 | 0.773 | 0.727 |
| synth_long_needle (300) | 0.843 | 0.723 | 0.827 | 0.940 | n/a | 0.583 | n/a | 0.333 | 0.270 | 1.000 | 1.000 |
| synth_multi_dependent (1346) | 0.424 | 0.435 | 0.467 | 0.591 | n/a | 0.444 | 0.344 | 0.277 | 0.333 | 0.728 | 0.771 |
| synth_options_xl (300) | 0.957 | 0.600 | 0.953 | 0.630 | n/a | 0.000 | 0.023 | 0.000 | 0.000 | 0.200 | 0.197 |
| synth_policy_biz (300) | 0.560 | 0.560 | 0.593 | 0.690 | n/a | 0.643 | 0.340 | 0.340 | 0.370 | 0.900 | 0.933 |
| tasksource_jev (1529) | 0.692 | 0.704 | 0.694 | 0.700 | n/a | 0.728 | 0.460 | 0.511 | 0.487 | 0.731 | 0.734 |
| typed_synth (2540) | 0.738 | 0.656 | 0.737 | 0.865 | n/a | 0.744 | 0.656 | 0.488 | 0.609 | 0.903 | 0.886 |
| **Overall (complete models only)** | **0.795** | **0.719** | **0.794** | **0.793** | partial 0.685 | **0.745** | partial 0.610 | **0.519** | **0.600** | **0.821** | **0.828** |
| **Overall without synth_options_xl (API limit: max 255 options)** | **0.794** | **0.720** | **0.793** | **0.794** | partial 0.685 | **0.749** | partial 0.613 | **0.522** | **0.603** | **0.824** | **0.832** |

Models: espresso-380m (R12 candidate, with decision rule); vela3-fast (base model, raw); espresso R2 candidate (earlier round); clef-flash (teacher); clef-text-0.6b; gliclass-v4-edge; gliner2.5-decide; laya; open-jev-deberta; Jev 1.14 (TypeSafe API); Gemini Decision 1 (Google API)

Hosted APIs (Jev 1.14, Gemini Decision 1) reject choice questions with more than 255 options; those items count as wrong in the first overall row (as any refused item does), and the second overall row leaves that suite out for every model. Jev and jevbench/tasksource_jev share TypeSafe's request format (possible home advantage). Gemini Decision 1 also refused 20 items with a safety block (HTTP 400: 11 tasksource_jev, 8 jevbench, 1 pi_detect); they count as wrong. Hosted runs 2026-11-17: Jev pinned jev-1.14.0, Gemini gemini-decision-1 (v1beta).

