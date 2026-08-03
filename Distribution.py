#! wget https://cdn.intra.42.fr/document/document/51890/leaves.zip
#! unzip leaves.zip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
import seaborn as sns


def dataset_analysis():
    if len(sys.argv) != 2:
        raise Exception("Invalid number of arguments.")
    my_dir = Path(sys.argv[1])
    dict_counts = {}
    sub_directories_count = 0
    for item in my_dir.iterdir():
        # print(item.name)
        if item.is_dir():
            sub_directories_count += 1
            sub_dir = Path(sys.argv[1] + "/" + item.name)
            JPG_files = list(sub_dir.glob("*.JPG"))
            dict_counts[item.name] = len(JPG_files)
    # print(dict_counts)
    series = pd.Series(dict_counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    c_palette = sns.color_palette("Spectral", as_cmap=False, n_colors=sub_directories_count)
    sns.barplot(x=series.values, y=series.index, hue=series.index, legend=False, palette=c_palette, ax=axes[0])
    axes[0].set_title("Data distribution (bar plot)")
    axes[0].set_ylabel("Category of the images")
    axes[1].set_title("Data distribution (pie plot)")
    axes[1] = plt.pie(series, colors=c_palette, labels=series.index, )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #! wget https://cdn.intra.42.fr/document/document/51890/leaves.zip
    try: 
        dataset_analysis()
    except Exception as e:
        print(f"Exception: {e}")
    