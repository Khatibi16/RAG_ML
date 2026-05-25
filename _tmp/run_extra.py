import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import io, json, time

def read(p):
    return io.open("_tmp/" + p, encoding="utf-8").read()

nb = json.load(io.open("project.ipynb", encoding="utf-8"))
codecells = [c for c in nb["cells"] if c["cell_type"] == "code"]

def cell(sub):
    for c in codecells:
        if sub in "".join(c["source"]):
            return "".join(c["source"])
    raise RuntimeError("cell not found: " + sub)

G = {"__name__": "__rag__"}

# Library cells (definitions only — no heavy top-level execution).
for marker in [
    "import hashlib",
    "class config:",
    "def load_triviaqa",
    "def chunk_corpus",
    "class BaseRetriever",
    "class Generator",
    "def normalize_answer",
    "class RAGPipeline",
    "def save_results",
    "PALETTE = {",
]:
    exec(cell(marker), G)

# New definitions (identical text to the notebook cells).
exec(read("cell_exp9_def.py"), G)
exec(read("cell_paired_def.py"), G)
exec(read("cell_fig_def.py"), G)

setup = G["setup"]
print(">>> setup()")
t0 = time.time()
questions, corpus_docs, generator = setup()
print(">>> setup done in %.1fs; %d questions, %d docs" %
      (time.time() - t0, len(questions), len(corpus_docs)))

print(">>> Experiment 9 (controlled distraction)")
t0 = time.time()
G["experiment_9_distraction"](questions, corpus_docs, generator)
print(">>> Exp9 done in %.1fs" % (time.time() - t0))

print(">>> Paired significance")
G["run_paired_significance"]()

print(">>> Figures")
G["fig11_distraction"]()
G["fig12_significance"]()
print(">>> ALL DONE")
