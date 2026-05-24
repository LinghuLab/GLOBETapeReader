default:
    just --list

# ── ND2 extraction ──────────────────────────────────────────────────
extract_nd2:
    python -m globepipeline.segmentation.extract_nd2 --config=config.yaml

# ── Fiber segmentation (GPU, pytc env) ──────────────────────────────
fiber_segmentation:
    python -m globepipeline.segmentation.fiber_segmentation --config=config.yaml

# ── Cell segmentation (GPU, microsam env) ───────────────────────────
cell_segmentation:
    python -m globepipeline.segmentation.cell_segmentation --config=config.yaml

# ── Skeletonization ─────────────────────────────────────────────────
fiber_skeleton TILE:
    python -m globepipeline.processing.skeletonize_fibers --config=config.yaml --tile={{TILE}}

# ── Full pipeline for a single tile ─────────────────────────────────
pipeline TILE ND2NAME:
    python -m globepipeline.run_pipeline --config=config.yaml --tile={{TILE}} --nd2-name={{ND2NAME}}

# ── Plot signals ────────────────────────────────────────────────────
plot_signals:
    python -m globepipeline.plot.plot_signals --config=config.yaml

# ── Run everything for a tile ───────────────────────────────────────
everything TILE ND2NAME:
    just fiber_segmentation
    just cell_segmentation
    just pipeline {{TILE}} {{ND2NAME}}
