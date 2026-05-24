"""Fiber segmentation using MedNeXt (pytorch_connectomics).

Usage: python -m globepipeline.segmentation.fiber_segmentation --config=config.yaml
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

from nicety.conf import get_conf
from globepipeline.utils import CHANNEL_PROFILES


def load_fiber_seg(pred_dir, tile_name, profile="standard"):
    """Load pre-computed fiber segmentation mask for a tile.

    Tries _prediction_fixed.tiff first, then _prediction.tiff.
    """
    ch_files = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_files"]
    fiber_ch_idx = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["fiber_channel"]
    fiber_suffix = ch_files[fiber_ch_idx]

    pred_path = os.path.join(pred_dir, f"{tile_name}_{fiber_suffix}_prediction_fixed.tiff")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pred_dir, f"{tile_name}_{fiber_suffix}_prediction.tiff")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Fiber segmentation not found in {pred_dir}/ "
            f"(tried {tile_name}_{fiber_suffix}_prediction_fixed.tiff and "
            f"{tile_name}_{fiber_suffix}_prediction.tiff)"
        )

    fiber_seg = tifffile.imread(pred_path)
    n_instances = len(np.unique(fiber_seg)) - 1
    print(f"  Fiber seg: {fiber_seg.shape}, {n_instances} instances from {pred_path}")
    return fiber_seg


def run_inference(conf, tile_name):
    """Run MedNeXt inference on a single tile.

    This generates the YAML config and calls pytorch_connectomics inference.
    Requires a GPU.
    """
    from omegaconf import OmegaConf

    tile_dir = conf.tile_dir
    fiber_seg_dir = conf.fiber_seg_dir
    profile = getattr(conf, "profile", "standard")
    ch_files = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_files"]
    fiber_ch_idx = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["fiber_channel"]
    fiber_suffix = ch_files[fiber_ch_idx]

    input_tiff = os.path.join(tile_dir, f"{tile_name}_{fiber_suffix}.tif")
    if not os.path.exists(input_tiff):
        raise FileNotFoundError(f"Fiber channel TIFF not found: {input_tiff}")

    output_name = f"{tile_name}_{fiber_suffix}_prediction.tiff"
    output_path = os.path.join(fiber_seg_dir, output_name)

    if os.path.exists(output_path) or os.path.exists(output_path.replace(".tiff", "_fixed.tiff")):
        print(f"  Cached: {output_path}")
        return

    # Build inference config
    base_yaml = conf.fiber_segmentation.base_yaml
    checkpoint = conf.fiber_segmentation.checkpoint

    base_cfg = OmegaConf.load(base_yaml)
    override = {
        "SYSTEM": {"NUM_GPUS": conf.num_gpus, "NUM_CPUS": conf.num_cpus},
        "INFERENCE": {
            "INPUT_PATH": tile_dir,
            "IMAGE_NAME": f"{tile_name}_{fiber_suffix}.tif",
            "OUTPUT_PATH": fiber_seg_dir,
            "OUTPUT_NAME": output_name,
            "SAMPLES_PER_BATCH": conf.fiber_segmentation.batch_size,
        },
    }
    cfg = OmegaConf.merge(base_cfg, override)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=fiber_seg_dir, delete=False) as f:
        OmegaConf.save(cfg, f.name)
        yaml_path = f.name

    try:
        _run_pytc(yaml_path, checkpoint)
    finally:
        os.unlink(yaml_path)

    print(f"  Inference done: {output_path}")


def _run_pytc(yaml_path, checkpoint_path):
    """Call pytorch_connectomics inference engine."""
    import argparse as _argparse
    from connectomics.utils.system import init_devices
    from connectomics.config import load_cfg
    from connectomics.engine import Trainer

    args = _argparse.Namespace(
        config_file=yaml_path,
        config_base=None,
        inference=True,
        distributed=False,
        checkpoint=checkpoint_path,
        manual_seed=None,
        local_world_size=1,
        local_rank=None,
        debug=False,
        opts=[],
    )
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    mode = "test"
    trainer = Trainer(cfg, device, mode, rank=args.local_rank, checkpoint=args.checkpoint)

    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func()
    else:
        trainer.run_chunk(mode)


def main(conf):
    tile = getattr(conf, "tile", None)
    if tile is None:
        # Auto-detect tiles from extracted TIFFs
        tile_dir = Path(conf.tile_dir)
        profile = getattr(conf, "profile", "standard")
        ch_files = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_files"]
        fiber_ch_idx = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["fiber_channel"]
        fiber_suffix = ch_files[fiber_ch_idx]
        tiles = sorted(set(
            p.name.replace(f"_{fiber_suffix}.tif", "")
            for p in tile_dir.glob(f"*_{fiber_suffix}.tif")
        ))
    else:
        tiles = [tile]

    os.makedirs(conf.fiber_seg_dir, exist_ok=True)

    for t in (pbar := tqdm(tiles)):
        pbar.set_description(t)
        print(f"\n{'='*60}")
        print(f"Fiber Segmentation: {t}")
        print(f"{'='*60}")
        run_inference(conf, t)


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
