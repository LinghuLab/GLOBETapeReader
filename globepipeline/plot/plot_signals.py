"""Plot extracted signal profiles per channel.

Usage: python -m globepipeline.plot.plot_signals --config=config.yaml
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pick import pick

from nicety.conf import get_conf
from globepipeline.utils import get_basename


def plot_signals(conf, basename):
    basename = ".".join(basename.split(".")[:-1]) if "." in basename else basename
    fname = os.path.join(conf.output_dir, f"{basename}_profiles.npz")
    if not os.path.exists(fname):
        print(f"Profile NPZ not found: {fname}")
        return

    data = np.load(fname, allow_pickle=True)

    # Plot each channel that has (N, 1000) shape
    for key in sorted(data.files):
        arr = data[key]
        if arr.ndim != 2 or arr.shape[1] < 100:
            continue
        if "smooth_split" in key:
            continue  # skip smooth split by default

        plt.figure(figsize=(10, 4))
        for i in range(min(arr.shape[0], 500)):
            plt.plot(arr[i], alpha=0.2, linewidth=0.5)

        plt.xlabel("Centerline Point")
        plt.ylabel("Signal Value")
        plt.title(f"{key} (n={arr.shape[0]} fibers) — {basename}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    conf = get_conf()

    title = "Select the dataset to plot"
    profiles = sorted(glob.glob(os.path.join(conf.output_dir, "*_profiles.npz")))
    if not profiles:
        print(f"No profile NPZ files found in {conf.output_dir}")
    else:
        names = [os.path.basename(x) for x in profiles]
        _, index = pick(names, title, indicator="=>")
        print(f"Selected: {names[index]}")
        plot_signals(conf, names[index].replace("_profiles.npz", ""))
