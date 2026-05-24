"""Vectorized signal extraction along fiber skeletons.

Usage: python -m globepipeline.processing.extract_signals --config=config.yaml --tile A1
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from scipy.interpolate import interpn

from nicety.conf import get_conf
from globepipeline.utils import CHANNEL_PROFILES


def extract_signals_vectorized(skeletons, raw_channels, cell_seg, fiber_seg, conf):
    """Extract signals from all channels along each skeleton (batched interpn)."""
    print(f"\n{'='*60}")
    print("STEP: Signal Extraction (vectorized)")
    print(f"{'='*60}")

    anisotropy = conf.anisotropy
    profile = getattr(conf, "profile", "standard")
    ch_names = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_names"]

    fid_list = list(skeletons.keys())
    if not fid_list:
        print("  No skeletons found - skipping signal extraction")
        return skeletons

    all_centerlines = [skeletons[fid]["centerline"] for fid in fid_list]
    n_pts_per_fiber = [len(cl) for cl in all_centerlines]
    all_points = np.concatenate(all_centerlines, axis=0)
    print(f"  Batched {len(fid_list)} fibers, {len(all_points)} total points")

    # Build grid coordinates once
    vol_shape = cell_seg.shape
    grid = [np.arange(vol_shape[i], dtype=float) * anisotropy[i] for i in range(3)]

    # Clamp OOB points to volume edges (avoids fill_value=0 artifacts)
    n_oob = 0
    for dim in range(3):
        lo, hi = 0.0, grid[dim][-1]
        oob_mask = (all_points[:, dim] < lo) | (all_points[:, dim] > hi)
        n_oob += oob_mask.sum()
        all_points[:, dim] = np.clip(all_points[:, dim], lo, hi)
    if n_oob > 0:
        print(f"  Clamped {n_oob} out-of-bounds skeleton coordinates to volume edges")

    extracted = {}
    for ch_idx, ch_name in ch_names.items():
        if ch_idx in raw_channels:
            print(f"  Extracting {ch_name} (linear)...")
            extracted[ch_name] = interpn(
                grid, raw_channels[ch_idx].astype(float), all_points,
                method="linear", bounds_error=False, fill_value=0
            )

    print(f"  Extracting cell_seg (nearest)...")
    extracted["cell_seg"] = interpn(
        grid, cell_seg.astype(float), all_points,
        method="nearest", bounds_error=False, fill_value=0
    ).astype(np.int32)

    print(f"  Extracting fiber_seg (nearest)...")
    extracted["fiber_seg"] = interpn(
        grid, fiber_seg.astype(float), all_points,
        method="nearest", bounds_error=False, fill_value=0
    ).astype(np.int32)

    seg_keys = {"cell_seg", "fiber_seg"}
    offset = 0
    for fid, n_pts in zip(fid_list, n_pts_per_fiber):
        signals = {}
        for key, vals in extracted.items():
            signals[key] = vals[offset:offset + n_pts]
            if key not in seg_keys:
                signals[key] = signals[key].astype(float)
        skeletons[fid]["signals"] = signals
        offset += n_pts

    print(f"  Extracted signals for {len(skeletons)} fibers")
    return skeletons


def load_raw_channels(tile_name, conf):
    """Load per-channel TIFFs for a tile."""
    tile_dir = Path(conf.tile_dir)
    profile = getattr(conf, "profile", "standard")
    ch_files = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_files"]
    ch_names = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_names"]

    raw_channels = {}
    for ch_idx, suffix in ch_files.items():
        fpath = tile_dir / f"{tile_name}_{suffix}.tif"
        if fpath.exists():
            raw_channels[ch_idx] = tifffile.imread(str(fpath))
            print(f"  Ch{ch_idx} ({ch_names[ch_idx]}): {raw_channels[ch_idx].shape}")
        else:
            print(f"  WARNING: Ch{ch_idx} ({ch_names[ch_idx]}) not found: {fpath}")
    return raw_channels


def main(conf):
    tile = getattr(conf, "tile", None)
    if tile is None:
        raise ValueError("Must specify --tile or set tile in config")

    # Load cached skeletons
    cache_dir = Path(conf.cache_dir)
    skel_cache = cache_dir / f"{tile}_skeletons.npz"
    if not skel_cache.exists():
        raise FileNotFoundError(f"Skeleton cache not found: {skel_cache}. Run skeletonize_fibers first.")

    cached = np.load(str(skel_cache), allow_pickle=True)
    skeletons = {}
    for fid, cl, ln in zip(cached["fiber_ids"], cached["centerlines"], cached["lengths"]):
        skeletons[int(fid)] = {"centerline": cl, "length_nm": float(ln)}

    # Load raw channels
    raw_channels = load_raw_channels(tile, conf)

    # Load cell seg
    cell_seg_path = cache_dir / f"{tile}_cell_seg.npz"
    if cell_seg_path.exists():
        cell_seg = np.load(str(cell_seg_path))["cell_seg"]
    else:
        print(f"  WARNING: No cell seg at {cell_seg_path}, using zeros")
        first_ch = list(raw_channels.values())[0]
        cell_seg = np.zeros(first_ch.shape, dtype=np.int32)

    # Load fiber seg
    from globepipeline.segmentation.fiber_segmentation import load_fiber_seg
    profile = getattr(conf, "profile", "standard")
    fiber_seg = load_fiber_seg(conf.fiber_seg_dir, tile, profile=profile)

    extract_signals_vectorized(skeletons, raw_channels, cell_seg, fiber_seg, conf)


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
