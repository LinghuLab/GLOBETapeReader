import os
import copy
import numpy as np
import warnings


CHANNEL_PROFILES = {
    "standard": {
        "channel_names": {0: "dapi", 1: "fiber", 2: "cfos", 3: "timestamp"},
        "fiber_channel": 1,
        "timestamp_channel": 3,
        "dapi_channel": 0,
        "gfp_channel": None,
        "channel_files": {
            0: "ch0_dapi",
            1: "ch1",
            2: "ch2_cfos",
            3: "ch3_timestamp",
        },
    },
    "fos-gfp": {
        "channel_names": {0: "dapi", 1: "gfp", 2: "cfos", 3: "fiber"},
        "fiber_channel": 3,
        "timestamp_channel": None,
        "dapi_channel": 0,
        "gfp_channel": 1,
        "channel_files": {
            0: "ch0_dapi",
            1: "ch1_gfp",
            2: "ch2_cfos",
            3: "ch3",
        },
    },
    "reordered": {
        "channel_names": {0: "dapi", 1: "fiber", 2: "cfos", 3: "timestamp"},
        "fiber_channel": 1,
        "timestamp_channel": 3,
        "dapi_channel": 0,
        "gfp_channel": None,
        "channel_files": {
            0: "ch0_dapi",
            1: "ch1",
            2: "ch2_cfos",
            3: "ch3_timestamp",
        },
    },
}


ND2_CHANNEL_PROFILES = {
    "standard": {
        0: "ch0_dapi",
        1: "ch1",
        2: "ch2_cfos",
        3: "ch3_timestamp",
    },
    "reordered": {
        0: "ch2_cfos",
        1: "ch1",
        2: "ch0_dapi",
        3: "ch3_timestamp",
    },
    "fos-gfp": {
        0: "ch0_dapi",
        1: "ch1_gfp",
        2: "ch2_cfos",
        3: "ch3",
    },
}


def preprocess_vol(vol):
    """Transpose raw volume from (P, Z, C, Y, X) or (Z, C, Y, X) to (C, Z, Y, X)."""
    assert vol.ndim == 4, f"Expected 4D image, got {vol.ndim}D"
    if vol.shape[1] != 4:
        warnings.warn(
            f"Expected 4 channels, got {vol.shape[1]} channels. "
            "This may lead to unexpected behavior."
        )
    # C, Z, Y, X
    vol = np.transpose(vol, (1, 0, 2, 3))
    return vol


def get_basename(path):
    return "-".join(os.path.basename(path).split("-")[:-1])


def get_config(conf):
    """Apply channel profile to config dict. `conf` is an OmegaConf DotDict or dict."""
    profile = conf.get("profile", "standard") if hasattr(conf, "get") else "standard"
    if profile in CHANNEL_PROFILES:
        for key, val in CHANNEL_PROFILES[profile].items():
            if hasattr(conf, key):
                setattr(conf, key, copy.deepcopy(val))
            elif isinstance(conf, dict):
                conf[key] = copy.deepcopy(val)
    return conf
