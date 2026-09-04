### Leaffliction

### Installation
0. Have `uv` installed.
1. For school, `export UV_PROJECT_ENVIRONMENT=~/goinfre/venvs/leaffliction MPLBACKEND=TkAgg`
2. What's your device? CUDA => `uv sync --extra=cuda`; NOT CUDA => `uv sync --extra=cpu`

### Running
1. What's your device? CUDA => `uv run train.py -d cuda <dir_of_imgs>`; NOT CUDA => `uv run train.py -d cpu <dir_of_imgs>`.

### Data attribution
In-the-field data from:
- **AppleLeaf9** (Yang, Duan & Wang, "Efficient Identification of Apple Leaf Diseases in the Wild Using Convolutional Neural Networks," Agronomy 12.11 (2022): 2784, https://doi.org/10.3390/agronomy12112784), source: https://github.com/JasonYangCode/AppleLeaf9, licensed CC BY 4.0.
- **GVLiD** (Shikalgar, Anisa; Savalkar, Ayush; Bhasme, Avishkar; Chavan, Snehal; Nikam, Vaishnavi (2026), "GVLiD: GrapeVine Leaf identification of the Diseases", Mendeley Data, V5, doi: 10.17632/wkymf8bhcg.5), licensed CC BY 4.0.
