# GLOBE Tape Reader pipeline

End-to-end pipeline for large-scale, brain-wide analysis of GLOBE recording from microscopy images.

## Key Features

| Feature | TapeReader | GLOBE-Tapereader |
|---|---|---|
| **Input** | Pre-split `.tif` files `(C,Z,Y,X)` | `.nd2` files; pipeline extracts tiles |
| **Fiber seg** | pytorch_connectomics inline + CLAHE + BCS watershed | MedNeXt model with built-in post-processing |
| **Cell seg** | micro-sam in same env | micro-sam in separate `microsam` env |
| **Signal extraction** | Per-fiber interpn | Vectorized batched interpn with OOB clamping |
| **Midpoint detection** | Single-pass Optuna `[0.45, 0.55]` | Two-pass: coarse grid `[0.3, 0.7]` + Optuna fine-tune |
| **Normalization** | Raw midpoint only | Raw + smoothed midpoint with 3 Pearson r metrics |
| **Validation** | Length, PCA, soma, one-per-soma | Same + divergent spline Z-span check |
| **Channel profiles** | Fixed 4-channel mapping | Configurable: `standard`, `reordered`, `fos-gfp` |
| **Export** | Nested dict NPZ | Flat array NPZ (analysis-ready) |

## Directory Structure

```
GLOBE-Tapereader/
├── config.yaml                          # Pipeline parameters
├── justfile                             # Task runner
├── README.md
└── globepipeline/
    ├── __init__.py
    ├── utils.py                         # Channel profiles, preprocessing
    ├── run_pipeline.py                  # Full pipeline orchestrator
    ├── segmentation/
    │   ├── __init__.py
    │   ├── extract_nd2.py               # ND2 → per-channel tile TIFFs
    │   ├── cell_segmentation.py         # micro-sam cell segmentation
    │   └── fiber_segmentation.py        # MedNeXt fiber segmentation
    ├── processing/
    │   ├── __init__.py
    │   ├── skeletonize_fibers.py        # PCA + spline skeletonization
    │   ├── extract_signals.py           # Vectorized signal extraction
    │   ├── normalize_signals.py         # Midpoint detection + recentering
    │   ├── validate_fibers.py           # Quality filtering
    │   └── export_profiles.py           # NPZ export
    └── plot/
        ├── __init__.py
        └── plot_signals.py              # Matplotlib signal visualization
```

## Quick Start

```bash
# 1. Extract tiles from ND2
just extract_nd2

# 2. Run fiber segmentation (pytc env, GPU)
just fiber_segmentation

# 3. Run cell segmentation (microsam env, GPU)
conda activate microsam
just cell_segmentation

# 4. Run full analysis pipeline for a tile
just pipeline A1 A1-2003

# Or run everything:
just everything A1 A1-2003
```

## Configuration

All parameters are in `config.yaml`. Key settings:

- **anisotropy**: Voxel size in nm `[Z, Y, X]` — from ND2 metadata
- **profile**: Channel mapping (`standard`, `reordered`, `fos-gfp`)
- **skeletonize_fibers**: Spline fitting parameters
- **normalize_signals**: Midpoint detection range and Optuna trials
- **validate_fibers**: Quality thresholds

## Output

Per-tile NPZ files with flat arrays:
- `fiber_ids`, `fiber_length_um`, `pca_linearity`, `z_span_um`
- `parent_cell_id`, `centroid_{z,y,x}_um`, `volume_um3`
- `midpoint_raw`, `midpoint_smooth`, `pearson_r_*`
- `Nissl`, `structural monomer`, `signal monomer`, `timestamp monomer` — (N, 1000) profiles
- `*_smooth_split` — profiles recentered at smoothed midpoint
