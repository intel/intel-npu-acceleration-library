#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import json
import glob
import os


df = None
for files in glob.glob("./leaderboard_*.json"):
    with open(files) as fp:
        prof = json.load(fp)
        date = datetime_object = datetime.datetime.strptime(
            prof["config"]["time"], "%Y-%m-%d_%H-%M-%S"
        )

        new_df = pd.DataFrame.from_records(prof["profiling"])
        new_df["date"] = date
        if df is None:
            df = new_df
        else:
            df = pd.concat([df, new_df], axis=0, join="outer")

df.pop("error")


col_to_str = {
    "model": "Model",
    "context_size": "Context Size",
    "tps": "Tokens / s",
    "prefill": "Prefill (s)",
    "intel_npu_acceleration_library": "Intel® NPU Acceleration Library enabled",
    "dtype": "Datatype",
}


def plot(df, x, y, hue="context_size", title=None, latest=True):

    filtered = df[(df["date"] == df["date"].max())] if latest else df

    plt.figure(figsize=(16, 9))
    ax = sns.barplot(filtered.dropna(), x=x, y=y, hue=hue)
    ax.set_xlabel(col_to_str[x])
    plt.xticks(rotation=45)
    if y == "prefill":
        ax.set_ylabel(f"Log {col_to_str[y]}")
        # ax.set_yscale('log')
    else:
        ax.set_ylabel(col_to_str[y])
    if title is None:
        title = f"{col_to_str[y]} vs {col_to_str[x]}"
    ax.set_title(title)
    # ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., title=col_to_str[hue])
    ax.legend(title=col_to_str[hue])
    filename = f"data/{x}_{y}_{hue}.png"
    os.makedirs("data", exist_ok=True)
    print(f"Save image {title} to {filename}")
    ax.get_figure().savefig(filename, bbox_inches="tight")


sns.color_palette("tab10")

plot(df[df["context_size"] == 128], "model", "tps", hue="dtype")
plot(df[df["context_size"] == 128], "model", "prefill", hue="dtype")

plot(df[df["intel_npu_acceleration_library"] == True], "model", "tps")
plot(df[df["intel_npu_acceleration_library"] == True], "model", "prefill")
