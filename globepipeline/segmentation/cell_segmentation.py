"""Cell segmentation using micro-sam (requires 'microsam' conda env).

Usage: python -m globepipeline.segmentation.cell_segmentation --config=config.yaml
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)
from nicety.conf import get_conf


def detect_tiles(tile_dir):
    """Auto-detect tile names from DAPI files in the tile directory."""
    tile_dir = Path(tile_dir)
    tiles = sorted(set(
        p.name.replace("_ch0_dapi.tif", "")
        for p in tile_dir.glob("*_ch0_dapi.tif")
    ))
    return tiles


def segment(vol, model_type):
    predictor, segmenter = get_predictor_and_segmenter(model_type)
    seg = automatic_instance_segmentation(
        predictor, segmenter, input_path=vol, verbose=True
    )
    return seg


def run_cell_seg(tile_name, tile_dir, output_dir, model_type="vit_b_lm"):
    """Run micro-sam cell segmentation on DAPI channel for one tile."""
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)

    dapi_path = tile_dir / f"{tile_name}_ch0_dapi.tif"
    output_path = output_dir / f"{tile_name}_cell_seg.npz"

    os.makedirs(output_dir, exist_ok=True)

    if output_path.exists():
        print(f"  Cached: {output_path}")
        return

    print(f"Loading DAPI: {dapi_path}")
    dapi = tifffile.imread(str(dapi_path))
    print(f"  Shape: {dapi.shape}, dtype: {dapi.dtype}, range: [{dapi.min()}-{dapi.max()}]")

    print(f"Running micro-sam ({model_type}) on full 3D volume...")
    seg = segment(dapi, model_type)
    seg = seg.astype(np.int32)
    n_labels = len(np.unique(seg)) - 1
    print(f"  Segmentation: {seg.shape}, {n_labels} cell instances")

    np.savez_compressed(str(output_path), cell_seg=seg)
    print(f"  Saved: {output_path}")


def main(conf):
    tile_dir = conf.tile_dir
    output_dir = conf.cache_dir
    model_type = getattr(conf.cell_segmentation, "model_type", "vit_b_lm")

    tile = getattr(conf, "tile", "all")
    if tile == "all":
        tiles = detect_tiles(tile_dir)
        if not tiles:
            print(f"ERROR: No DAPI tiles found in {tile_dir}")
            return
        print(f"Auto-detected {len(tiles)} tiles: {', '.join(tiles)}")
    else:
        tiles = [tile]

    for t in (pbar := tqdm(tiles)):
        pbar.set_description(t)
        print(f"\n{'='*60}")
        print(f"Cell Segmentation: {t}")
        print(f"{'='*60}")
        run_cell_seg(t, tile_dir, output_dir, model_type=model_type)


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
