import io, json, sys

NB = "project.ipynb"

def read(p):
    return io.open("_tmp/" + p, encoding="utf-8").read()

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}

nb = json.load(io.open(NB, encoding="utf-8"))
cells = nb["cells"]

def find(substr, start=0):
    for i in range(start, len(cells)):
        if cells[i]["cell_type"] == "code" and substr in "".join(cells[i]["source"]):
            return i
    raise RuntimeError("not found: " + substr)

# Idempotency: bail if already inserted.
if any("experiment_9_distraction" in "".join(c["source"]) for c in cells if c["cell_type"]=="code"):
    print("Cells already present — nothing to insert.")
    sys.exit(0)

# 1) Add DISTRACTION_N_VALUES to the config cell.
ci = find("class config:")
src = cells[ci]["source"]
anchor = "    DISTRACTOR_SWEEP_VALUES = [0, 500, 2000, 5000]\n"
for j, line in enumerate(src):
    if line == anchor:
        src.insert(j + 1,
            "\n    # ── Controlled distraction (Experiment 9) ───────────────\n"
            "    # Hold retrieval recall at 1 (gold chunk always present) and\n"
            "    # vary the number of non-gold distractor chunks beside it.\n"
            "    DISTRACTION_N_VALUES = [0, 1, 2, 4, 8]\n")
        break
else:
    raise RuntimeError("config anchor not found")

# 2) Exp 9 cells, after the experiment_8 code cell.
i8 = find("def experiment_8_distractor_sweep")
exp9_md = (
    "## 17b. Experiment 9 — Controlled distraction (recall held at 1)\n\n"
    "Experiments 5 and 8 cannot isolate *generator distraction*: in Exp 5 RAG\n"
    "almost never hurts, and in Exp 8 the topic-agnostic distractors never\n"
    "out-rank the gold pages, so Recall@5 is flat. This experiment guarantees\n"
    "the answer is present (Recall@k = 1) and varies only the amount of\n"
    "competing *non-gold* retrieved context, so any EM/F1 drop is pure\n"
    "distraction rather than a retrieval miss.\n"
)
exp9_code = read("cell_exp9_def.py") + \
    "\n\nexp9_results = experiment_9_distraction(questions, corpus_docs, generator)\n"

# 3) Paired significance cells, right after Exp 9.
paired_md = (
    "## 17c. Paired significance tests\n\n"
    "All conditions are evaluated on the **same** questions, so the correct\n"
    "comparison is *paired*. The marginal bootstrap CIs plotted elsewhere\n"
    "answer \"is this one number reliable?\"; they systematically *understate*\n"
    "the significance of a within-question A-vs-B contrast because the\n"
    "(large) question-difficulty variance is shared by both systems and\n"
    "cancels under pairing. We report **McNemar's exact test** on the EM\n"
    "hit/miss table and a **paired bootstrap** on the per-question EM and F1\n"
    "deltas for every contrast that shares a question set.\n"
)
paired_code = read("cell_paired_def.py") + \
    "\n\npaired_significance = run_paired_significance()\n"

new_after_exp = [
    md_cell(exp9_md), code_cell(exp9_code),
    md_cell(paired_md), code_cell(paired_code),
]
cells[i8 + 1:i8 + 1] = new_after_exp

# 4) Figures for Exp 9 + significance, after the fig10 code cell.
i10 = find("def fig10_distractor_sweep")
fig_src = read("cell_fig_def.py")
fig11_code = (fig_src[:fig_src.index("def fig12_significance")].rstrip()
              + "\n\n\nfig11_distraction()\n")
fig12_code = ("def fig12_significance" + fig_src.split("def fig12_significance", 1)[1].rstrip()
              + "\n\n\nfig12_significance()\n")
cells[i10 + 1:i10 + 1] = [code_cell(fig11_code), code_cell(fig12_code)]

json.dump(nb, io.open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Inserted cells. Total cells now:", len(cells))
