"""Full fiber analysis pipeline.

Usage:
    python -m globepipeline.run_pipeline --config=config.yaml --tile A1 --nd2-name A1-2003
    python -m globepipeline.run_pipeline --config=config.yaml --tile A1 --nd2-name A1-2003 --steps skeletonize,extract,normalize,validate,export
"""

import os
import numpy as np
from pathlib import Path

from nicety.conf import get_conf

from globepipeline.segmentation.fiber_segmentation import load_fiber_seg
from globepipeline.segmentation.cell_segmentation import run_cell_seg
from globepipeline.processing.skeletonize_fibers import generate_fiber_skeletons
from globepipeline.processing.extract_signals import (
    extract_signals_vectorized,
    load_raw_channels,
)
from globepipeline.processing.normalize_signals import normalize_all_signals
from globepipeline.processing.validate_fibers import validate_fibers
from globepipeline.processing.export_profiles import export_profiles


def load_cell_segmentation(output_path):
    """Load pre-computed micro-sam cell segmentation."""
    print("\nSTEP: Load Cell Segmentation")

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Cell segmentation not found: {output_path}\n"
            f"Run cell_segmentation.py in the 'microsam' conda env first."
        )

    cell_seg = np.load(output_path)["cell_seg"]
    n_labels = len(np.unique(cell_seg)) - 1
    print(f"  Loaded: {output_path}")
    print(f"  Shape: {cell_seg.shape}, {n_labels} total 3D cell labels")
    return cell_seg


def run_pipeline(conf, tile_name, nd2_name, steps=None):
    """Run the full pipeline on a single tile."""
    all_steps = ["cell_seg", "skeletonize", "extract", "normalize", "validate", "export"]
    if steps is None:
        steps = all_steps

    print(f"\nFIBER PIPELINE: {nd2_name} / {tile_name}")

    cache_dir = Path(conf.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    profile = getattr(conf, "profile", "standard")

    print(f"\nLoading fiber segmentation...")
    fiber_seg = load_fiber_seg(conf.fiber_seg_dir, tile_name, profile=profile)
    fiber_seg_nz = fiber_seg.shape[0]

    cell_seg = None
    cell_seg_cache = cache_dir / f"{tile_name}_cell_seg.npz"
    if cell_seg_cache.exists():
        cell_seg = load_cell_segmentation(str(cell_seg_cache))
    else:
        print(f"\n  WARNING: Cell seg not found at {cell_seg_cache}")
        print(f"  Run: conda activate microsam && python -m globepipeline.segmentation.cell_segmentation --config=config.yaml --tile {tile_name}")

    if cell_seg is None:
        print("  WARNING: No cell segmentation available, creating zeros")
        cell_seg = np.zeros_like(fiber_seg, dtype=np.int32)

    skeletons = None
    skel_cache = cache_dir / f"{tile_name}_skeletons.npz"
    if "skeletonize" in steps:
        skeletons, _ = generate_fiber_skeletons(conf, tile_name=tile_name)
    elif skel_cache.exists():
        cached = np.load(str(skel_cache), allow_pickle=True)
        skeletons = {}
        volumes = cached.get("volumes", [0.0] * len(cached["fiber_ids"]))
        for fid, cl, ln, vol in zip(cached["fiber_ids"], cached["centerlines"], cached["lengths"], volumes):
            skeletons[int(fid)] = {"centerline": cl, "length_nm": float(ln), "volume_um3": float(vol)}
        print(f"\n  Loaded cached skeletons: {len(skeletons)} fibers")

    if skeletons is None:
        raise RuntimeError("No skeletons available. Run 'skeletonize' step first.")

    if len(skeletons) == 0:
        print(f"\n  No fibers detected in tile {tile_name} — skipping remaining steps")
        return None

    if "extract" in steps:
        raw_channels = load_raw_channels(tile_name, conf)
        skeletons = extract_signals_vectorized(skeletons, raw_channels, cell_seg, fiber_seg, conf)

    normalized = None
    if "normalize" in steps:
        normalized = normalize_all_signals(skeletons, conf)

    if normalized is None:
        normalized = skeletons

    validated = None
    if "validate" in steps:
        validated = validate_fibers(normalized, conf, fiber_seg_nz=fiber_seg_nz)

    if validated is None:
        validated = normalized

    npz_path = None
    if "export" in steps:
        npz_path = export_profiles(validated, nd2_name, tile_name, conf, cell_seg=cell_seg)

    print(f"\nPIPELINE COMPLETE: {nd2_name} / {tile_name}")
    return npz_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fiber analysis pipeline")
    parser.add_argument("--tile", required=True, help="Tile name (e.g., A1)")
    parser.add_argument("--nd2-name", required=True, help="ND2 file name (e.g., A1-2003)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated steps: cell_seg,skeletonize,extract,normalize,validate,export")
    args, remaining = parser.parse_known_args()

    conf = get_conf(remaining)

    steps = args.steps.split(",") if args.steps else None
    run_pipeline(conf, args.tile, args.nd2_name, steps=steps)
