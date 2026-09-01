# ! wget https://cdn.intra.42.fr/document/document/51890/leaves.zip
# ! unzip leaves.zip
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# import argparse
from pathlib import Path
import seaborn as sns

from utils.dataset import discover_class_images


def dataset_analysis():
    if len(sys.argv) != 2:
        raise Exception("Invalid number of arguments.")
    my_dir = Path(sys.argv[1])
    class_images = discover_class_images(my_dir)
    set_plants = set()
    dict_counts = {}
    for class_name, images in class_images.items():
        set_plants.add(class_name.split(sep='_')[0])
        dict_counts[class_name] = len(images)
    series = pd.Series(dict_counts)
    fig, axes = plt.subplots(len(set_plants), 2,
                             figsize=(5 * len(set_plants), 10))
    for i, plant in enumerate(set_plants):
        plant_series = series[series.index.str.startswith(plant)]
        c_palette = sns.color_palette("Spectral", as_cmap=False,
                                      n_colors=len(plant_series))
        # print(plant_series)
        sns.barplot(x=plant_series.values, y=plant_series.index,
                    hue=plant_series.index,
                    legend=False, palette=c_palette, ax=axes[i][0])
        axes[i][0].set_title(f"Data distribution for {plant} (bar plot)")
        axes[i][0].set_ylabel("Category of the images")
        axes[i][1].set_title(f"Data distribution for {plant} (pie plot)")
        axes[i][1].pie(plant_series, colors=c_palette,
                       labels=plant_series.index)
    plt.tight_layout()
    plt.savefig
    plt.show()


if __name__ == '__main__':
    # ! wget https://cdn.intra.42.fr/document/document/51890/leaves.zip
    try:
        dataset_analysis()
    except Exception as e:
        print(f"Exception: {e}")
