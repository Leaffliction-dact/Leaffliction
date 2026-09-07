### Leaffliction

### Installation

0. Have `uv` installed: https://docs.astral.sh/uv/getting-started/installation/
1. For school 42, `export UV_PROJECT_ENVIRONMENT=~/goinfre/venvs/leaffliction MPLBACKEND=TkAgg`
2. What's your device? CUDA => `uv sync --extra=cuda`; NOT CUDA => `uv sync --extra=cpu`

### Running

#### In general

What's your device? CUDA => `uv run train.py -d cuda <params>`; not CUDA => it will automatically use the CPU.

#### Specifics

The most interesting training you can do here is an incrementally partial unfreeze/finetune of the resnet18.

This sequence of events assumes a 42 computer. The images are structured as such:

```
parent
     \Apple_Black_rot
	                \image ..(1).JPG
					 image ..(2).JPG
					 ...
      Apple_healthy
      Apple_rust
      Apple_scab
      Grape_Black_rot
      Grape_Esca
      Grape_healthy
      Grape_spot
```

The names of the files don't matter but they all should be JPGs. The names of the class folders matter, down to casing.

1. Run the base thing, only tuning it to the PlatVillage dataset:

```
uv run train.py -a resnet18 -e 10 \
-c ~/goinfre/_prepared_pv_test \
-m step1_resnet18.pt \
<parent from before>
```

Note that perhaps an `-n 100` to `-n 500` might be necessarry to cut down on processing times. Same with the epoch count (`-e`) - you might want to adjust it to be higher if you can afford it.

Note the `-c` flag. It indicates the directory where the pre-processed images are cached into; for subsequent runs and iterations a `-p` to load from there directly is recommended to reduce on preparation times.

The `-m` flag indicates the filename of the model to be saved.

2. Before step three we actually need our external images prepared to the format used by 42:

```
uv run prepare_field_data.py -s appleleaf9 ~/goinfre/AppleLeaf9
uv run prepare_field_data.py -s gvlid ~/goinfre/GVLiD/GVLiD/DATASET
```

See to `-o` for a custom output dir. Default is `~/goinfre/field_data`. You only need to run this comand once per the dataset you'll be working with; it links the apple and grape images obtained from external sources into a 42-style folder structure.

3. Run the field data through, still only on the head of the model:

```
uv run train.py -a resnet18 -n 100 -e 10 \
-c ~/goinfre/_prepared_field_test \
-F step1_resnet18.pt \
-m step2_resnet18.pt \
<your field data links parent>
```

`-n 100` is manageable. `-n 500` is insanely slow on 42's machines.

4. Finally, run the unfrozen layer 4 step:

```
uv run train.py -a resnet18 -e 10 \
-p ~/goinfre/_prepared_field_test \
-F step2_resnet18.pt \
-u layer4 \
-m step3_resnet18.pt
```

Worthy of attention: the `-u` parameter also can be set to `layer3` for deeper unfreezing. The lack of `-n` is due to using the already prepared images from the command before that.

Ta-da! You're now the proud owner of a finetuned plantvillage + real world data model for determining 4 types of apples and 4 types of grapes. Go ahead, give it a shot:

```
uv run predict.py -h
```

*You can do it*

### Data attribution

In-the-field data from:

- **AppleLeaf9** (Yang, Duan & Wang, "Efficient Identification of Apple Leaf Diseases in the Wild Using Convolutional Neural Networks," Agronomy 12.11 (2022): 2784, https://doi.org/10.3390/agronomy12112784), source: https://github.com/JasonYangCode/AppleLeaf9, licensed CC BY 4.0.
- **GVLiD** (Shikalgar, Anisa; Savalkar, Ayush; Bhasme, Avishkar; Chavan, Snehal; Nikam, Vaishnavi (2026), "GVLiD: GrapeVine Leaf identification of the Diseases", Mendeley Data, V5, doi: 10.17632/wkymf8bhcg.5), licensed CC BY 4.0.
