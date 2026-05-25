# Sanity sweep: try alternative no-context prompts and report EM/F1 per template.
NO_CONTEXT_ALTS = {
    "default (current)": (
        "Answer the following question with a short phrase.\n\n"
        "Question: {question}\n"
        "Short answer:"
    ),
    "qa_cue":  "Q: {question}\nA:",
    "trivia":  "Trivia: {question}",
    "polite":  (
        "Please answer the following trivia question.\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
}

no_rag_sensitivity = {}
golds = [q["answers"] for q in questions]
for name, tmpl in NO_CONTEXT_ALTS.items():
    prompts_alt = [tmpl.format(question=q["question"]) for q in questions]
    preds_alt   = generator.generate(prompts_alt)
    em = [exact_match(p, g) for p, g in zip(preds_alt, golds)]
    f1 = [token_f1(p, g)    for p, g in zip(preds_alt, golds)]
    em_mean, em_lo, em_hi = bootstrap_ci(em)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1)
    no_rag_sensitivity[name] = {
        "template": tmpl,
        "em": {"mean": em_mean, "lo": em_lo, "hi": em_hi, "n": len(em)},
        "f1": {"mean": f1_mean, "lo": f1_lo, "hi": f1_hi, "n": len(f1)},
        "sample": [
            {"q": questions[i]["question"], "gold": golds[i][:2], "pred": preds_alt[i]}
            for i in range(min(5, len(questions)))
        ],
    }
    logger.info(
        "%-18s  EM %.3f [%.3f-%.3f]   F1 %.3f [%.3f-%.3f]",
        name, em_mean, em_lo, em_hi, f1_mean, f1_lo, f1_hi,
    )

save_results("exp5_no_rag_prompt_sensitivity", no_rag_sensitivity)

best_name, best = max(
    no_rag_sensitivity.items(), key=lambda kv: kv[1]["em"]["mean"]
)
print(f"\nBest no-context prompt: {best_name!r}  (EM={best['em']['mean']:.3f},  F1={best['f1']['mean']:.3f})")
print("If this is materially above the default, replace")
print("config.PROMPT_TEMPLATES['no_context'] with this template and re-run experiment_5.")