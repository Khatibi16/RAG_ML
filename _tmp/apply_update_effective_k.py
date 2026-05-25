"""Idempotent in-place update for the effective-k reporting change.

Replaces only the cells that have changed since the previous pass:
  * cell 11  (Generator)
  * cell 15  (RAGPipeline)
  * exp6 oracle code cell (find by 'def experiment_6_oracle')
Leaves cell order and counts untouched.
"""
import json
from pathlib import Path

NB_PATH = Path(r"C:\Users\Filip\Documents\ml-project\RAG_ML\project.ipynb")
TMP     = Path(r"C:\Users\Filip\Documents\ml-project\RAG_ML\_tmp")


def split_to_source(text: str):
    return text.splitlines(keepends=True) if text else []


def find_cell_idx(cells, needle):
    for i, c in enumerate(cells):
        if needle in "".join(c["source"]):
            return i
    raise RuntimeError(f"Could not find cell containing {needle!r}")


def replace_cell(cells, idx, source_text):
    cells[idx]["source"] = split_to_source(source_text)
    if cells[idx]["cell_type"] == "code":
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None


def main():
    nb    = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    replace_cell(
        cells,
        find_cell_idx(cells, "class Generator:"),
        (TMP / "new_cell_11_generator.py").read_text(encoding="utf-8"),
    )
    replace_cell(
        cells,
        find_cell_idx(cells, "class RAGPipeline"),
        (TMP / "new_cell_15_pipeline.py").read_text(encoding="utf-8"),
    )
    replace_cell(
        cells,
        find_cell_idx(cells, "def experiment_6_oracle"),
        (TMP / "new_cell_exp6_oracle.py").read_text(encoding="utf-8"),
    )

    NB_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Updated {NB_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
