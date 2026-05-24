"""Fiber validation and filtering.

Usage: python -m globepipeline.processing.validate_fibers --config=config.yaml --tile A1
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from sklearn.decomposition import PCA

from nicety.conf import get_conf
from globepipeline.utils import CHANNEL_PROFILES


def evaluate_fiber(data, conf):
    """Compute validation metrics for a single fiber."""
    centerline = data["centerline"]
    profile = getattr(conf, "profile", "standard")
    prof = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])

    cl_orig = data.get("centerline_original", centerline)
    ext = conf.skeletonize_fibers.extrapolate
    n = len(cl_orig)
    total_range = ext[1] - ext[0]
    if total_range <= 0:
        total_range = 1.0
    core_start = int(round((0 - ext[0]) / total_range * n))
    core_end = int(round((1 - ext[0]) / total_range * n))
    core_start = max(0, min(n, core_start))
    core_end = max(core_start + 1, min(n, core_end))
    core = cl_orig[core_start:core_end]
    if len(core) < 2:
        core = cl_orig
    length_nm = np.sum(np.linalg.norm(np.diff(core, axis=0), axis=1)) if len(core) > 1 else 0.0
    length_um = length_nm / 1000.0

    center = np.mean(centerline, axis=0)
    radius = np.max(np.linalg.norm(centerline - center, axis=1))
    if radius > 0:
        pts_norm = (centerline - center) / radius
        pca = PCA(n_components=3)
        pca.fit(pts_norm)
        pca_ratio = pca.explained_variance_ratio_[0]
    else:
        pca_ratio = 0.0

    cell_labels = data["signals"].get("cell_seg", np.array([0]))
    cell_label = int(mode(cell_labels, keepdims=False).mode)

    dapi_ch_name = prof["channel_names"][prof["dapi_channel"]]
    mean_soma = np.mean(data["signals"].get(dapi_ch_name, np.array([0])))

    mean_soma_gfp = 0.0
    if prof.get("gfp_channel") is not None:
        gfp_ch_name = prof["channel_names"][prof["gfp_channel"]]
        mean_soma_gfp = np.mean(data["signals"].get(gfp_ch_name, np.array([0])))

    z_span_um = (core[:, 0].max() - core[:, 0].min()) / 1000.0 if len(core) > 0 else 0.0

    return {
        "length_um": length_um,
        "pca_ratio": pca_ratio,
        "cell_label": cell_label,
        "mean_soma": mean_soma,
        "mean_soma_gfp": mean_soma_gfp,
        "z_span_um": z_span_um,
    }


def validate_fibers(normalized, conf, fiber_seg_nz=54):
    """Filter fibers by quality criteria."""
    print("\nSTEP: Fiber Validation")

    val_cfg = conf.validate_fibers
    anisotropy = conf.anisotropy

    valid = {}
    stats = {"total": len(normalized), "short": 0, "low_pca": 0, "no_soma": 0, "divergent": 0}

    max_z_um = anisotropy[0] / 1000.0 * fiber_seg_nz * 3

    for fid, data in tqdm(normalized.items(), desc="  Validating"):
        metrics = evaluate_fiber(data, conf)

        if metrics["z_span_um"] > max_z_um:
            stats["divergent"] += 1
            is_valid = False
        elif metrics["length_um"] < val_cfg.thres_length_um:
            stats["short"] += 1
            is_valid = False
        elif metrics["pca_ratio"] < val_cfg.thres_pca_ratio:
            stats["low_pca"] += 1
            is_valid = False
        elif metrics["mean_soma"] < val_cfg.thres_mean_soma:
            stats["no_soma"] += 1
            is_valid = False
        else:
            is_valid = True

        data.update(metrics)
        data["is_valid"] = is_valid
        valid[fid] = data

    n_valid = sum(1 for d in valid.values() if d["is_valid"])
    print(f"  Total fibers: {stats['total']}")
    print(f"  Divergent spline: {stats['divergent']}")
    print(f"  Too short (<{val_cfg.thres_length_um}µm): {stats['short']}")
    print(f"  Low PCA ratio: {stats['low_pca']}")
    print(f"  No soma signal: {stats['no_soma']}")
    print(f"  Valid fibers: {n_valid}/{len(valid)}")

    if getattr(val_cfg, "one_per_soma", False):
        cell_to_fibers = {}
        for fid, data in valid.items():
            if data["is_valid"] and data["cell_label"] != 0:
                cell_to_fibers.setdefault(data["cell_label"], []).append(fid)
        for cell_id, fids in cell_to_fibers.items():
            if len(fids) > 1:
                best = max(fids, key=lambda f: valid[f]["length_um"])
                for f in fids:
                    if f != best:
                        valid[f]["is_valid"] = False
        n_valid_after = sum(1 for d in valid.values() if d["is_valid"])
        print(f"  After one-per-soma: {n_valid_after}/{n_valid}")

    return valid


def main(conf):
    print("Run via: python -m globepipeline.run_pipeline --config=config.yaml --tile A1")


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
