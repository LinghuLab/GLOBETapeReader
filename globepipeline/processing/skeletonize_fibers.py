"""Fiber skeletonization via PCA + spline fitting.

Usage: python -m globepipeline.processing.skeletonize_fibers --config=config.yaml
"""

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from scipy.interpolate import splprep, splev, interp1d
from sklearn.decomposition import PCA
from skimage.measure import regionprops

from nicety.conf import get_conf
from globepipeline.segmentation.fiber_segmentation import load_fiber_seg


def fit_cylinder_spline_pca(
    points,
    manual_z_scale=1.0,
    percentile_fit=(0.0, 1.0),
    spline_smoothing=None,
    npoints_geodesic=1000,
):
    """Fit a spline to a point cloud using PCA + arc-length reparameterization.

    Returns (evaluate_spline_fn, total_arc_length).
    """
    assert len(percentile_fit) == 2

    points = points * np.array([manual_z_scale, 1.0, 1.0])
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    if radius == 0:
        radius = 1.0
    points = (points - center) / radius

    pca = PCA(n_components=3)
    pca.fit(points)
    points_pca = pca.transform(points)

    t_values = points_pca[:, 0]
    t_sorted, sorted_unique_indices = np.unique(t_values, return_index=True)
    points_pca_sorted = points_pca[sorted_unique_indices]

    t_range = t_sorted.max() - t_sorted.min()
    if t_range == 0:
        t_range = 1.0
    t_normalized = (t_sorted - t_sorted.min()) / t_range

    used = (t_normalized >= percentile_fit[0]) & (t_normalized <= percentile_fit[1])
    points_pca_sorted = points_pca_sorted[used]
    t_normalized = t_normalized[used]

    if len(points_pca_sorted) < 4:
        raise ValueError(f"Too few points ({len(points_pca_sorted)}) for spline fitting")

    tck, u = splprep(
        [points_pca_sorted[:, 0], points_pca_sorted[:, 1], points_pca_sorted[:, 2]],
        u=t_normalized, s=spline_smoothing,
    )

    u_dense = np.linspace(0, 1, npoints_geodesic)
    spline_points_dense = np.array(splev(u_dense, tck)).T
    diff = np.diff(spline_points_dense, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    arc_lengths_normalized = arc_lengths / arc_lengths[-1] if arc_lengths[-1] > 0 else arc_lengths

    arc_to_param = interp1d(arc_lengths_normalized, u_dense, kind="linear",
                            bounds_error=False, fill_value="extrapolate")

    def evaluate_spline(s_values):
        """Evaluate spline at arc-length parameter values in [0, 1]."""
        u_values = arc_to_param(s_values)
        spline_pca = np.array(splev(u_values, tck)).T
        points_xyz = pca.inverse_transform(spline_pca)
        points_xyz = points_xyz * radius + center
        points_xyz = points_xyz * np.array([1.0 / manual_z_scale, 1.0, 1.0])
        return points_xyz

    return evaluate_spline, arc_lengths[-1]


def skeletonize_single_fiber(fiber_mask_crop, bbox, anisotropy, skel_cfg):
    """Skeletonize one fiber. Returns (centerline, length) or (None, 0)."""
    points = np.argwhere(fiber_mask_crop) * np.array(anisotropy)
    if len(points) < 10:
        return None, 0

    try:
        spline_fn, _ = fit_cylinder_spline_pca(
            points,
            manual_z_scale=skel_cfg.manual_z_scale,
            percentile_fit=skel_cfg.percentile_fit,
            spline_smoothing=skel_cfg.spline_smoothing,
        )

        core_pts = spline_fn(np.linspace(0, 1, 200))
        total_length = np.sum(np.linalg.norm(np.diff(core_pts, axis=0), axis=1))

        if total_length > 500_000:
            return None, 0

        extrapolate = skel_cfg.extrapolate
        centerline = spline_fn(
            np.linspace(extrapolate[0], extrapolate[1], skel_cfg.num_centerline_points)
        )
        centerline[:, 0] += bbox[0] * anisotropy[0]
        centerline[:, 1] += bbox[2] * anisotropy[1]
        centerline[:, 2] += bbox[4] * anisotropy[2]
        return centerline, total_length
    except Exception:
        return None, 0


def generate_fiber_skeletons(conf, tile_name=None):
    """Skeletonize all fibers in a tile. Returns (skeletons_dict, fiber_seg)."""
    profile = getattr(conf, "profile", "standard")
    anisotropy = conf.anisotropy

    fiber_seg = load_fiber_seg(conf.fiber_seg_dir, tile_name, profile=profile)
    n_z = fiber_seg.shape[0]

    voxel_vol_um3 = (anisotropy[0] / 1000.0) * (anisotropy[1] / 1000.0) * (anisotropy[2] / 1000.0)

    fiber_ids = np.unique(fiber_seg)
    fiber_ids = fiber_ids[fiber_ids != 0]
    print(f"  Found {len(fiber_ids)} fiber instances")

    props = regionprops(fiber_seg)
    fiber_data = []
    fiber_voxel_counts = {}

    for prop in props:
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        crop = fiber_seg[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        mask = (crop == prop.label)
        fiber_data.append((prop.label, mask, (bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5])))
        fiber_voxel_counts[prop.label] = int(prop.area)

    def process_fiber(label_id, mask, bbox):
        centerline, length = skeletonize_single_fiber(mask, bbox, anisotropy, conf.skeletonize_fibers)
        return label_id, centerline, length

    n_jobs = getattr(conf, "num_cpus", -1)
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(process_fiber)(lid, mask, bbox) for lid, mask, bbox in fiber_data
        ),
        total=len(fiber_data),
        desc="  Skeletonizing",
    ))

    skeletons = {}
    failed = 0
    for label_id, centerline, length in results:
        if centerline is not None:
            skeletons[label_id] = {
                "centerline": centerline,
                "length_nm": length,
                "volume_um3": fiber_voxel_counts[label_id] * voxel_vol_um3,
            }
        else:
            failed += 1

    print(f"  Successfully skeletonized: {len(skeletons)}/{len(fiber_data)} ({failed} failed)")

    cache_dir = Path(conf.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skel_cache = cache_dir / f"{tile_name}_skeletons.npz"
    np.savez_compressed(str(skel_cache),
        fiber_ids=np.array(list(skeletons.keys())),
        centerlines=[s["centerline"] for s in skeletons.values()],
        lengths=[s["length_nm"] for s in skeletons.values()],
        volumes=[s["volume_um3"] for s in skeletons.values()],
    )
    print(f"  Cached: {skel_cache}")

    return skeletons, fiber_seg


def main(conf):
    tile = getattr(conf, "tile", None)
    if tile is None:
        raise ValueError("Must specify --tile or set tile in config")
    generate_fiber_skeletons(conf, tile_name=tile)


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
