"""
Regenerate exp5_no_rag.json + exp5_summary.json from the generation cache
using the current canonical no-context template (qa_cue: "Q: {question}\nA:").

The saved Exp 5 files were produced with the older long-form template
(EM=0.02) before the config switched to qa_cue (EM=0.07). All 100 qa_cue
predictions are already in data/cache/generation_cache.json from §14's
sensitivity sweep, so we can rebuild the JSONs without running any model.
"""
import hashlib
import json
import pickle
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

QA_CUE_TEMPLATE = "Q: {question}\nA:"
MAX_INPUT_TOKENS = 1024
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42


def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(c for c in t if c not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred, golds):
    n = normalize_answer(pred)
    return float(any(n == normalize_answer(g) for g in golds))


def token_f1(pred, golds):
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        nc = sum(common.values())
        if nc == 0:
            continue
        p = nc / len(pt)
        r = nc / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def bootstrap_ci(scores, n_bootstrap=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def main():
    with open(CACHE / "triviaqa_rc_n100_w5_wd2000.pkl", "rb") as f:
        questions, _ = pickle.load(f)
    with open(CACHE / "generation_cache.json", "r", encoding="utf-8") as f:
        gen_cache = json.load(f)

    predictions = []
    prompts = []
    for q in questions:
        prompt = QA_CUE_TEMPLATE.format(question=q["question"])
        key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if key not in gen_cache:
            raise SystemExit(f"Missing cache for qid={q['qid']}")
        prompts.append(prompt)
        predictions.append(gen_cache[key])

    golds = [q["answers"] for q in questions]

    em_scores = [exact_match(p, g) for p, g in zip(predictions, golds)]
    f1_scores = [token_f1(p, g) for p, g in zip(predictions, golds)]

    em_mean, em_lo, em_hi = bootstrap_ci(em_scores)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_scores)

    metrics = {
        "em": {"mean": em_mean, "lo": em_lo, "hi": em_hi, "n": len(em_scores)},
        "f1": {"mean": f1_mean, "lo": f1_lo, "hi": f1_hi, "n": len(f1_scores)},
    }

    per_example = []
    for q, pred, gold, prompt in zip(questions, predictions, golds, prompts):
        per_example.append({
            "qid": q["qid"],
            "question": q["question"],
            "gold": gold,
            "prediction": pred,
            "em": exact_match(pred, gold),
            "f1": token_f1(pred, gold),
            "recall": 0.0,
            "n_retrieved": 0,
            "prompt": prompt,
            "top_chunk": "",
        })

    # Prompt-token stats (so the saved file matches the notebook schema).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    raw_lens = [len(tok.encode(p, truncation=False, add_special_tokens=True))
                for p in prompts]
    timing = {
        "retrieval_s": 0.0,
        "generation_s": 0.0,
        "total_s": 0.0,
        "n_questions": len(questions),
        "cache_hits": len(questions),
        "cache_misses": 0,
        "prompt_tokens_mean": float(np.mean(raw_lens)),
        "prompt_tokens_max": int(np.max(raw_lens)),
        "prompt_tokens_budget": MAX_INPUT_TOKENS,
        "prompts_truncated": int(sum(1 for L in raw_lens if L > MAX_INPUT_TOKENS)),
        "n_prompts": len(raw_lens),
    }

    no_rag_payload = {
        "predictions": predictions,
        "gold_answers": golds,
        "metrics": metrics,
        "per_example": per_example,
        "timing": timing,
    }

    with open(RESULTS / "exp5_no_rag.json", "w", encoding="utf-8") as f:
        json.dump(no_rag_payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote exp5_no_rag.json   EM={em_mean:.3f}  F1={f1_mean:.3f}")

    # Rebuild exp5_summary.json using the existing rag result + new no_rag.
    with open(RESULTS / "exp5_rag.json", "r", encoding="utf-8") as f:
        rag_payload = json.load(f)

    rag_per = {ex["qid"]: ex for ex in rag_payload["per_example"]}
    no_rag_per = {ex["qid"]: ex for ex in per_example}

    delta = []
    for qid, rag_ex in rag_per.items():
        if qid not in no_rag_per:
            continue
        no_rag_ex = no_rag_per[qid]
        delta.append({
            "qid": qid,
            "question": rag_ex["question"],
            "gold": rag_ex["gold"],
            "rag_pred": rag_ex["prediction"],
            "no_rag_pred": no_rag_ex["prediction"],
            "rag_em": rag_ex["em"],
            "no_rag_em": no_rag_ex["em"],
            "rag_f1": rag_ex["f1"],
            "no_rag_f1": no_rag_ex["f1"],
            "recall": rag_ex.get("recall", 0),
            "rag_helps": int(rag_ex["em"] > no_rag_ex["em"]),
            "rag_hurts": int(rag_ex["em"] < no_rag_ex["em"]),
        })

    helps = sum(d["rag_helps"] for d in delta)
    hurts = sum(d["rag_hurts"] for d in delta)
    ties = len(delta) - helps - hurts

    summary = {
        "rag": {"em": rag_payload["metrics"]["em"], "f1": rag_payload["metrics"]["f1"]},
        "no_rag": {"em": metrics["em"], "f1": metrics["f1"]},
        "delta_counts": {"helps": helps, "hurts": hurts, "ties": ties},
        "delta_analysis": delta,
    }
    with open(RESULTS / "exp5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote exp5_summary.json  helps={helps} hurts={hurts} ties={ties}")


if __name__ == "__main__":
    main()
