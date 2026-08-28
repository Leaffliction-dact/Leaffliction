### Leaffliction

### Installation
0. Have `uv` installed.
1. For school, `export UV_PROJECT_ENVIRONMENT=~/goinfre/venvs/leaffliction`
2. What's your device? CUDA => `uv sync --extra=cuda`; NOT CUDA => `uv sync --extra=cpu`

### Running
1. What's your device? CUDA => `uv run train.py -d cuda <dir_of_imgs>`; NOT CUDA => `uv run train.py -d cpu <dir_of_imgs>`.
