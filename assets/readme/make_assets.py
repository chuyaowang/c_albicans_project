"""Generates the README's compressed images from the (gitignored) thesis_figures/
sources. Resizes each to a max width of 1600px, keeping PNG format so diagram
text and label overlays stay sharp.

Run from the pyg env (Pillow):
    conda run -n pyg python assets/readme/make_assets.py
"""
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = REPO_ROOT / "thesis_figures"
OUT_DIR = Path(__file__).resolve().parent
MAX_WIDTH = 1600

SOURCES = [
    (FIGURES / "flow.png", "01_pipeline_flow.png"),
    (FIGURES / "assembled_fig_4dyes.png", "02_raw_data.png"),
    (FIGURES / "generalization" / "generalization_before_after.png", "03_generalization.png"),
    (FIGURES / "Node_type_GCN.png", "04_gcn_architecture.png"),
    (FIGURES / "gcn_cell_graph" / "CET112_HWP1Cal610_6h_GoC_01_DIC.png", "05_cell_merge_graph.png"),
    (FIGURES / "gcn_nodetype_edge_interpret" / "example_fold1.png", "06_edge_prediction_example.png"),
    (FIGURES / "node_type_classification" / "node_type_labels.png", "07_node_type_labels.png"),
    (FIGURES / "gcn_node_interpret" / "example_fold1.png", "08_node_type_prediction_example.png"),
]


def main() -> None:
    for src, name in SOURCES:
        im = Image.open(src)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        if im.width > MAX_WIDTH:
            im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
        out_path = OUT_DIR / name
        im.save(out_path, optimize=True)
        print(f"{name}: {im.width}x{im.height}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()