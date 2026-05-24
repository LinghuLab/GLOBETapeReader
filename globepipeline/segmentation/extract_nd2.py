"""Extract tiles from ND2 file with metadata.

Usage: python -m globepipeline.segmentation.extract_nd2 --config=config.yaml
"""

import nd2
import numpy as np
import tifffile
import json
from pathlib import Path

from nicety.conf import get_conf
from globepipeline.utils import ND2_CHANNEL_PROFILES


def extract_all_tiles(conf, tile_indices=None):
    """
    Extract tiles from ND2 file.

    Returns:
        List of tile names extracted.
    """
    nd2_path = conf.nd2_path
    output_dir = Path(conf.tile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = getattr(conf, "profile", "standard")
    all_channels = getattr(conf, "all_channels", True)

    print(f"Opening ND2 file: {nd2_path}")
    with nd2.ND2File(nd2_path) as f:
        print(f"\nFile structure:")
        print(f"  Shape: {f.shape}")
        print(f"  Dimensions: {f.sizes}")

        n_positions = f.sizes.get('P', 1)
        n_z = f.sizes['Z']
        n_channels = f.sizes['C']

        voxel_size = f.voxel_size()
        pixel_size_um = {
            'x': float(voxel_size.x),
            'y': float(voxel_size.y),
            'z': float(voxel_size.z),
        }
        print(f"  Pixel size: x={pixel_size_um['x']:.4f}, y={pixel_size_um['y']:.4f}, z={pixel_size_um['z']:.4f} µm")

        # Stage positions
        tile_info = {}
        for loop in f.experiment:
            if loop.type == 'XYPosLoop':
                for i, point in enumerate(loop.parameters.points):
                    tile_info[i] = {
                        'name': point.name,
                        'stage_x': float(point.stagePositionUm.x),
                        'stage_y': float(point.stagePositionUm.y),
                        'stage_z': float(point.stagePositionUm.z),
                    }
                break

        if not tile_info:
            nd2_stem = Path(nd2_path).stem
            tile_info[0] = {
                'name': nd2_stem,
                'stage_x': 0.0,
                'stage_y': 0.0,
                'stage_z': 0.0,
            }
            print("  No XYPosLoop found — treating as single-position ND2")

        print(f"\nFound {len(tile_info)} tile positions:")
        for i, info in tile_info.items():
            print(f"  Tile {i} ({info['name']}): ({info['stage_x']:.2f}, {info['stage_y']:.2f}, {info['stage_z']:.2f}) µm")

        if tile_indices is None:
            tile_indices = list(range(n_positions))
        for idx in tile_indices:
            if idx >= n_positions:
                raise ValueError(f"Tile {idx} out of range (0-{n_positions-1})")

        print(f"\nLoading full ND2 array into memory...")
        full_data = f.asarray()
        if 'P' not in f.sizes:
            full_data = full_data[np.newaxis, ...]
        print(f"  Array shape: {full_data.shape}, dtype: {full_data.dtype}")

        channel_suffixes = ND2_CHANNEL_PROFILES.get(profile, ND2_CHANNEL_PROFILES["standard"])
        fiber_ch = 3 if profile == "fos-gfp" else 1
        if all_channels:
            channels_to_extract = list(range(n_channels))
        else:
            channels_to_extract = [fiber_ch]

        tile_names = []

        for tile_idx in tile_indices:
            info = tile_info[tile_idx]
            tile_name = info['name']
            tile_names.append(tile_name)

            print(f"\n{'='*60}")
            print(f"Tile {tile_idx}: {tile_name}")
            print(f"  Stage: ({info['stage_x']:.2f}, {info['stage_y']:.2f}, {info['stage_z']:.2f}) µm")

            for ch_idx in channels_to_extract:
                volume = full_data[tile_idx, :, ch_idx, :, :]
                suffix = channel_suffixes.get(ch_idx, f"ch{ch_idx}")
                tiff_path = output_dir / f"{tile_name}_{suffix}.tif"
                tifffile.imwrite(str(tiff_path), volume, compression='zlib')
                print(f"  Ch{ch_idx} ({suffix}): {volume.shape}, [{volume.min()}-{volume.max()}] → {tiff_path}")

            metadata = {
                'tile_idx': tile_idx,
                'tile_name': tile_name,
                'shape': list(full_data[tile_idx, :, 0, :, :].shape),
                'n_slices': n_z,
                'n_channels': n_channels,
                'pixel_size_um': pixel_size_um,
                'stage_position_um': {
                    'x': info['stage_x'],
                    'y': info['stage_y'],
                    'z': info['stage_z'],
                },
            }
            json_path = output_dir / f"{tile_name}_metadata.json"
            with open(str(json_path), 'w') as jf:
                json.dump(metadata, jf, indent=2)

        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE: {len(tile_names)} tiles, {len(channels_to_extract)} channels each")
        print(f"{'='*60}")
        for name in tile_names:
            print(f"  {name}")

        return tile_names


if __name__ == "__main__":
    conf = get_conf()
    tiles = getattr(conf, "tiles", None)
    extract_all_tiles(conf, tile_indices=tiles)
