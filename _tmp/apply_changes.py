"""Apply all extension changes to project.ipynb.

This script:
1. Replaces cells 03 (config), 09 (retrievers), 10 (generator MD),
   11 (generator code), 15 (RAGPipeline).
2. Inserts new cells after the existing exp5/prompt-sensitivity section:
   - Exp 6 (oracle) markdown + code
   - Exp 7 (rerank) markdown + code
   - Exp 8 (distractor sweep) markdown + code
3. Inserts new figure cells (fig8/9/10) after fig7.
4. Renumbers section headers in markdown cells.
5. Updates the figures-table in the analysis-figures markdown cell.
6. Writes the result back to project.ipynb.
"""

import json
import uuid
from pathlib import Path

NB_PATH = Path(r"C:\Users\Filip\Documents\ml-project\RAG_ML\project.ipynb")
TMP     = Path(r"C:\Users\Filip\Documents\ml-project\RAG_ML\_tmp")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def split_to_source(text: str) -> list:
    """Split text into a list of newline-terminated strings, matching
    Jupyter's storage format (last line has no trailing newline)."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


def new_cell(cell_type: str, source_text: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "id":        uuid.uuid4().hex[:12],
        "metadata":  {},
        "source":    split_to_source(source_text),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"]         = []
    return cell


def find_cell_idx(cells, needle, start=0):
    for i in range(start, len(cells)):
        src = "".join(cells[i]["source"])
        if needle in src:
            return i
    raise RuntimeError(f"Could not find cell containing {needle!r}")


def replace_cell(cells, idx, source_text, cell_type=None):
    cells[idx]["source"] = split_to_source(source_text)
    if cell_type:
        cells[idx]["cell_type"] = cell_type
    # Reset outputs for code cells so stale plots don't confuse the reader.
    if cells[idx]["cell_type"] == "code":
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ---- 1. Replace cell 03 (config) ----
    idx = find_cell_idx(cells, "class config:")
    replace_cell(cells, idx, read_text(TMP / "new_cell_03_config.py"))

    # ---- 2. Replace cell 09 (retrievers — adds RerankRetriever) ----
    idx = find_cell_idx(cells, "class BM25Retriever")
    replace_cell(cells, idx, read_text(TMP / "new_cell_09_retrievers.py"))

    # ---- 3. Replace cell 10 (Generator MD) ----
    idx = find_cell_idx(cells, "## 5. Generator")
    replace_cell(cells, idx, read_text(TMP / "new_cell_10_generator_md.md"),
                 cell_type="markdown")

    # ---- 4. Replace cell 11 (Generator code) ----
    idx = find_cell_idx(cells, "class Generator:")
    replace_cell(cells, idx, read_text(TMP / "new_cell_11_generator.py"))

    # ---- 5. Replace cell 15 (RAGPipeline) ----
    idx = find_cell_idx(cells, "class RAGPipeline")
    replace_cell(cells, idx, read_text(TMP / "new_cell_15_pipeline.py"))

    # ---- 6. Insert new experiments AFTER cell 32 (interpretation of
    #         no-RAG prompt sensitivity), which has '## 14' / 'Interpretation'.
    insert_after = find_cell_idx(cells, "Interpretation")
    new_cells = [
        new_cell("markdown", read_text(TMP / "new_cell_exp6_oracle_md.md")),
        new_cell("code",     read_text(TMP / "new_cell_exp6_oracle.py")),
        new_cell("markdown", read_text(TMP / "new_cell_exp7_rerank_md.md")),
        new_cell("code",     read_text(TMP / "new_cell_exp7_rerank.py")),
        new_cell("markdown", read_text(TMP / "new_cell_exp8_sweep_md.md")),
        new_cell("code",     read_text(TMP / "new_cell_exp8_sweep.py")),
    ]
    cells[insert_after + 1 : insert_after + 1] = new_cells

    # ---- 7. Update the "Analysis — figures" markdown cell to renumber
    #         to "## 18" and include the three new figures in its table.
    idx = find_cell_idx(cells, "## 15. Analysis — figures")
    analysis_md = (
        "## 18. Analysis — figures\n"
        "\n"
        "Generate the figures used in the report from the cached JSON results.\n"
        "Each figure is saved at 300 dpi to `figures/` and displayed inline below.\n"
        "\n"
        "| File | Description |\n"
        "|------|-------------|\n"
        "| `fig1_retriever_comparison.png` | EM + F1 for BM25 / TF-IDF / Dense |\n"
        "| `fig2_chunk_size.png`           | EM / F1 / Recall@k vs chunk size |\n"
        "| `fig3_k_values.png`             | EM / F1 / Recall@k vs k |\n"
        "| `fig4_prompt_template.png`      | EM + F1 for concise vs instructed |\n"
        "| `fig5_rag_vs_no_rag.png`        | EM + F1 grouped bar: RAG vs baseline |\n"
        "| `fig6_error_analysis.png`       | Pie + scatter: when does RAG help? |\n"
        "| `fig7_qualitative.png`          | Example table of RAG-helps vs RAG-hurts |\n"
        "| `fig8_oracle.png`               | No-RAG / RAG / Oracle bars (Exp 6) |\n"
        "| `fig9_rerank.png`               | Dense vs Dense + Reranker (Exp 7) |\n"
        "| `fig10_distractor_sweep.png`    | EM / F1 / Recall@k vs n_distractors (Exp 8) |\n"
    )
    replace_cell(cells, idx, analysis_md, cell_type="markdown")

    # ---- 8. Append fig8 / fig9 / fig10 cells AFTER fig7 ----
    idx = find_cell_idx(cells, "fig7_qualitative()")
    new_figure_cells = [
        new_cell("code", read_text(TMP / "new_cell_fig8_oracle.py")),
        new_cell("code", read_text(TMP / "new_cell_fig9_rerank.py")),
        new_cell("code", read_text(TMP / "new_cell_fig10_sweep.py")),
    ]
    cells[idx + 1 : idx + 1] = new_figure_cells

    # ---- 9. Write back ----
    NB_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {NB_PATH}")
    print(f"Total cells now: {len(cells)}")


if __name__ == "__main__":
    main()
