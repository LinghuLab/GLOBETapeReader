"""Signal normalization: tail cropping, midpoint detection, recentering.

Usage: python -m globepipeline.processing.normalize_signals --config=config.yaml --tile A1
"""

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

import optuna

from nicety.conf import get_conf
from globepipeline.utils import CHANNEL_PROFILES


def get_tail_crop_points(signal, extrapolate, gaussian_filter_ratio):
    """Find crop points by detecting where signal drops off at the tails."""
    smoothed = gaussian_filter1d(signal, int(gaussian_filter_ratio * len(signal)))
    num_points = signal.shape[0]
    start = int((0 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)
    stop = int((1 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)

    while (start > 0 and signal[start] > 0
           and smoothed[start - 1] < smoothed[start]):
        start -= 1
    while (stop < len(signal) - 1 and signal[stop] > 0
           and smoothed[stop + 1] < smoothed[stop]):
        stop += 1

    return start, stop


def apply_midpoint_scale(data, index, scale, num_points, calculate_pearson=False):
    """Split data at midpoint, optionally scale halves, resample to fixed length."""
    dtype = data.dtype
    is_float = np.issubdtype(dtype, np.floating)
    index = int(data.shape[0] * index)
    x, y = data[:index][::-1], data[index:]

    if len(x) < 2 or len(y) < 2:
        if calculate_pearson:
            return data, 0.0
        return data

    if scale < 0:
        x = x[:max(1, int((1 - abs(scale)) * x.shape[0]))]
    elif scale > 0:
        y = y[:max(1, int((1 - abs(scale)) * y.shape[0]))]

    half = num_points // 2
    if len(x) < 2 or len(y) < 2:
        if calculate_pearson:
            return data, 0.0
        return data

    x = interp1d(np.linspace(0, 1, x.shape[0]), x,
                  kind="linear" if is_float else "nearest"
                  )(np.linspace(0, 1, half)).astype(dtype)
    y = interp1d(np.linspace(0, 1, y.shape[0]), y,
                  kind="linear" if is_float else "nearest"
                  )(np.linspace(0, 1, half)).astype(dtype)

    result = np.concatenate([x[::-1], y])
    if calculate_pearson:
        if np.std(x) == 0 or np.std(y) == 0:
            return result, 0.0
        pearson = pearsonr(x, y).statistic
        return result, pearson
    return result


def get_midpoint_scale(signal, midpoint_range, scale_range, num_trials, num_points):
    """Two-pass midpoint search: coarse grid [0.3,0.7] then Optuna fine-tune."""
    coarse_pts = np.linspace(0.3, 0.7, 41)
    best_r, best_frac = -2.0, 0.5
    for frac in coarse_pts:
        _, r = apply_midpoint_scale(signal, frac, 0.0, num_points, calculate_pearson=True)
        if r > best_r:
            best_r, best_frac = r, frac

    fine_lo = max(0.25, best_frac - 0.05)
    fine_hi = min(0.75, best_frac + 0.05)

    def objective(trial):
        index = trial.suggest_float("index", fine_lo, fine_hi)
        scale = trial.suggest_float("scale", scale_range[0], scale_range[1])
        _, pearson = apply_midpoint_scale(signal, index, scale, num_points, calculate_pearson=True)
        return -pearson

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    return study.best_params["index"], study.best_params["scale"]


def get_geodesic(skel, center_idx):
    """Compute signed geodesic distance from center point along skeleton."""
    dist = np.cumsum(np.linalg.norm(np.diff(skel, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    dist -= dist[center_idx]
    return dist


def normalize_single_fiber(skel_data, conf):
    """Crop tails, find midpoint (raw + smoothed), recenter all channels."""
    signals = skel_data["signals"]
    centerline = skel_data["centerline"]
    skel_cfg = conf.skeletonize_fibers
    norm_cfg = conf.normalize_signals
    n_pts = skel_cfg.num_centerline_points

    profile = getattr(conf, "profile", "standard")
    prof = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["standard"])
    fiber_ch_name = prof["channel_names"][prof["fiber_channel"]]
    timestamp_ch_name = (prof["channel_names"][prof["timestamp_channel"]]
                         if prof.get("timestamp_channel") is not None else None)

    if fiber_ch_name not in signals:
        return None

    try:
        start, stop = get_tail_crop_points(
            signals[fiber_ch_name], skel_cfg.extrapolate, norm_cfg.gaussian_filter_ratio
        )
    except Exception:
        start, stop = 0, len(signals[fiber_ch_name]) - 1

    if stop - start < 10:
        return None

    centerline = centerline[start:stop + 1]
    signals = {k: v[start:stop + 1] for k, v in signals.items()}

    has_timestamp = (timestamp_ch_name is not None
                     and timestamp_ch_name in signals
                     and len(signals[timestamp_ch_name]) > 10)

    if has_timestamp:
        try:
            midpoint_raw, scale_raw = get_midpoint_scale(
                signals[timestamp_ch_name],
                norm_cfg.midpoint_range,
                norm_cfg.scale_range,
                norm_cfg.num_trials,
                n_pts,
            )
        except Exception:
            midpoint_raw, scale_raw = 0.5, 0.0
    else:
        midpoint_raw, scale_raw = 0.5, 0.0

    if has_timestamp:
        smooth_sigma = max(1, int(getattr(norm_cfg, "smooth_sigma_ratio", 0.02) * len(signals[timestamp_ch_name])))
        ts_smoothed = gaussian_filter1d(signals[timestamp_ch_name].astype(float), smooth_sigma)
        try:
            midpoint_smooth, scale_smooth = get_midpoint_scale(
                ts_smoothed,
                norm_cfg.midpoint_range,
                norm_cfg.scale_range,
                norm_cfg.num_trials,
                n_pts,
            )
        except Exception:
            midpoint_smooth, scale_smooth = 0.5, 0.0
    else:
        midpoint_smooth, scale_smooth = 0.5, 0.0

    if has_timestamp:
        _, pearson_r_raw = apply_midpoint_scale(
            signals[timestamp_ch_name], midpoint_raw, scale_raw, n_pts, calculate_pearson=True)
        _, pearson_r_smooth_on_raw = apply_midpoint_scale(
            signals[timestamp_ch_name], midpoint_smooth, scale_smooth, n_pts, calculate_pearson=True)
        _, pearson_r_smooth = apply_midpoint_scale(
            ts_smoothed, midpoint_smooth, scale_smooth, n_pts, calculate_pearson=True)
    else:
        pearson_r_raw = 0.0
        pearson_r_smooth_on_raw = 0.0
        pearson_r_smooth = 0.0

    recentered = {}
    for key, sig in signals.items():
        recentered[key] = apply_midpoint_scale(sig, midpoint_raw, scale_raw, n_pts)

    recentered_smooth = {}
    for key, sig in signals.items():
        recentered_smooth[key] = apply_midpoint_scale(sig, midpoint_smooth, scale_smooth, n_pts)

    skel_z = apply_midpoint_scale(centerline[:, 0].astype(float), midpoint_raw, scale_raw, n_pts)
    skel_y = apply_midpoint_scale(centerline[:, 1].astype(float), midpoint_raw, scale_raw, n_pts)
    skel_x = apply_midpoint_scale(centerline[:, 2].astype(float), midpoint_raw, scale_raw, n_pts)
    centerline_recentered = np.stack([skel_z, skel_y, skel_x], axis=-1)

    geodesic = get_geodesic(centerline_recentered, n_pts // 2)

    return {
        "signals": recentered,
        "signals_smooth_split": recentered_smooth,
        "centerline": centerline_recentered,
        "geodesic": geodesic,
        "midpoint_raw": midpoint_raw,
        "scale_raw": scale_raw,
        "midpoint_smooth": midpoint_smooth,
        "scale_smooth": scale_smooth,
        "pearson_r_raw": pearson_r_raw,
        "pearson_r_smooth_on_raw": pearson_r_smooth_on_raw,
        "pearson_r_smooth": pearson_r_smooth,
    }


def normalize_all_signals(skeletons, conf):
    """Normalize signals for all fibers."""
    print("\nSTEP: Signal Normalization")

    results = {}
    failed = 0
    for fid in tqdm(skeletons, desc="  Normalizing"):
        result = normalize_single_fiber(skeletons[fid], conf)
        if result is not None:
            results[fid] = result
            results[fid]["length_nm"] = skeletons[fid]["length_nm"]
            results[fid]["volume_um3"] = skeletons[fid].get("volume_um3", 0.0)
            results[fid]["centerline_original"] = skeletons[fid]["centerline"]
        else:
            failed += 1

    print(f"  Normalized: {len(results)}/{len(skeletons)} ({failed} failed)")
    return results


def main(conf):
    print("Run via: python -m globepipeline.run_pipeline --config=config.yaml --tile A1")


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
