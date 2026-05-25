## 14. Prompt sensitivity of the No-RAG (parametric) baseline

The No-RAG arm of Experiment 5 measures what Flan-T5-base produces with
no retrieved context — its "parametric" answer. A natural concern is that
this baseline is sensitive to the exact wording of the no-context prompt:
a permissive prompt could score higher than an over-restrictive one, and
the choice would directly affect any "RAG vs No-RAG" conclusion. This
section quantifies that sensitivity by running four reasonable
no-context formulations on the same 100 questions — complementing the
RAG-side prompt ablation in Experiment 4 with a No-RAG-side equivalent.

The alternatives compared:

| Name | Template |
|------|----------|
| `default (current)` | `Answer the following question with a short phrase.\n\nQuestion: ...\nShort answer:` |
| `qa_cue` | `Q: ...\nA:` — Flan-T5's canonical QA cue |
| `trivia` | `Trivia: ...` — Flan-T5 has seen many `task: input` prompts during pre-training |
| `polite` | `Please answer the following trivia question.\n\nQuestion: ...\nAnswer:` |

If the spread across alternatives is small, the No-RAG baseline is
robust and can be reported as a genuine parametric-knowledge floor; if
the spread is large, the choice of canonical prompt would have to be
justified more carefully. We then adopt the best-scoring alternative as
the canonical no-context prompt for Experiment 5.