"""Export intensity profiles and metadata as flat NPZ.

Usage: python -m globepipeline.processing.export_profiles --config=config.yaml --tile A1
"""

import os
import numpy as np

from nicety.conf import get_conf
from globepipeline.utils import CHANNEL_PROFILES


def export_profiles(validated, nd2_name, tile_name, conf, cell_seg=None):
    """Save per-fiber scalar metrics and (N, 1000) intensity profiles to NPZ."""
    print("\nSTEP: Intensity Profile Export")

    output_dir = conf.output_dir
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{nd2_name}_{tile_name}_profiles.npz")

    profile = getattr(conf, "profile", "standard")
    ch_names = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])["channel_names"]
    n_pts = conf.skeletonize_fibers.num_centerline_points

    fids_sorted = sorted(validated.keys())
    fiber_ids = np.array(fids_sorted, dtype=np.int32)

    profiles = {}
    for ch_idx, ch_name in ch_names.items():
        ch_profiles = []
        for fid in fids_sorted:
            sig = validated[fid]["signals"].get(ch_name)
            if sig is not None:
                ch_profiles.append(sig)
            else:
                ch_profiles.append(np.zeros(n_pts))
        profiles[ch_name] = np.array(ch_profiles, dtype=np.float32)

    profiles_smooth = {}
    for ch_idx, ch_name in ch_names.items():
        ch_profiles = []
        for fid in fids_sorted:
            sig = validated[fid].get("signals_smooth_split", {}).get(ch_name)
            if sig is not None:
                ch_profiles.append(sig)
            else:
                ch_profiles.append(np.zeros(n_pts))
        profiles_smooth[ch_name + "_smooth_split"] = np.array(ch_profiles, dtype=np.float32)

    fiber_length_um = np.array([validated[fid]["length_um"] for fid in fids_sorted], dtype=np.float32)
    pca_linearity = np.array([validated[fid]["pca_ratio"] for fid in fids_sorted], dtype=np.float32)
    z_span_um = np.array([validated[fid]["z_span_um"] for fid in fids_sorted], dtype=np.float32)
    parent_cell_id = np.array([validated[fid]["cell_label"] for fid in fids_sorted], dtype=np.int32)
    volume_um3 = np.array([validated[fid].get("volume_um3", 0.0) for fid in fids_sorted], dtype=np.float32)

    midpoint_raw = np.array([validated[fid].get("midpoint_raw", 0.5) for fid in fids_sorted], dtype=np.float32)
    midpoint_smooth = np.array([validated[fid].get("midpoint_smooth", 0.5) for fid in fids_sorted], dtype=np.float32)
    pearson_r_raw = np.array([validated[fid].get("pearson_r_raw", 0.0) for fid in fids_sorted], dtype=np.float32)
    pearson_r_smooth_on_raw = np.array([validated[fid].get("pearson_r_smooth_on_raw", 0.0) for fid in fids_sorted], dtype=np.float32)
    pearson_r_smooth = np.array([validated[fid].get("pearson_r_smooth", 0.0) for fid in fids_sorted], dtype=np.float32)

    centroid_z = np.empty(len(fids_sorted), dtype=np.float32)
    centroid_y = np.empty(len(fids_sorted), dtype=np.float32)
    centroid_x = np.empty(len(fids_sorted), dtype=np.float32)
    for i, fid in enumerate(fids_sorted):
        cl = validated[fid].get("centerline_original", validated[fid]["centerline"])
        mid = len(cl) // 2
        centroid_z[i] = cl[mid, 0] / 1000.0
        centroid_y[i] = cl[mid, 1] / 1000.0
        centroid_x[i] = cl[mid, 2] / 1000.0

    total_cells = 0
    if cell_seg is not None:
        total_cells = int(len(np.unique(cell_seg)) - 1)
    fiber_plus_neurons = len(np.unique(parent_cell_id[parent_cell_id > 0]))

    np.savez_compressed(npz_path,
        fiber_ids=fiber_ids,
        nd2_name=nd2_name,
        tile_name=tile_name,
        total_cells=total_cells,
        fiber_plus_neurons=fiber_plus_neurons,
        fiber_length_um=fiber_length_um,
        pca_linearity=pca_linearity,
        z_span_um=z_span_um,
        parent_cell_id=parent_cell_id,
        centroid_z_um=centroid_z,
        centroid_y_um=centroid_y,
        centroid_x_um=centroid_x,
        volume_um3=volume_um3,
        midpoint_raw=midpoint_raw,
        midpoint_smooth=midpoint_smooth,
        pearson_r_raw=pearson_r_raw,
        pearson_r_smooth_on_raw=pearson_r_smooth_on_raw,
        pearson_r_smooth=pearson_r_smooth,
        **profiles,
        **profiles_smooth,
    )

    print(f"  Saved {len(fiber_ids)} fiber profiles ({n_pts} points each)")
    print(f"  Recording density: {fiber_plus_neurons}/{total_cells} Fiber+ neurons ({100*fiber_plus_neurons/max(total_cells,1):.1f}%)")
    print(f"  Channels (raw split): {list(ch_names.values())}")
    print(f"  Channels (smooth split): {list(profiles_smooth.keys())}")
    print(f"  → {npz_path}")
    return npz_path


def main(conf):
    print("Run via: python -m globepipeline.run_pipeline --config=config.yaml --tile A1")


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
