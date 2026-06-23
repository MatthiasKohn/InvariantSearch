# IVSN transformation-invariance experiments

This repository contains the Invariant Visual Search Network (IVSN) code and an
experiment for measuring robustness to rotations, scaling, shifts, skew, noise,
and blur. The experiment supports the original VGG backbone as well as several
Gist-based feature extractors.

## Installation

Python 3.8 or newer is recommended. Create and activate a virtual environment,
then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiment imports `Gist` with:

```python
from gist import Gist
```

`gist` is a project-specific model implementation, not part of Python's standard
library. Install the repository/package that provides this class, or add its
directory to `PYTHONPATH`, before running one of the Gist models. The plain VGG
model does not use a Gist checkpoint, although `models.py` currently imports the
class when the model module is loaded.

Main third-party libraries:

- PyTorch and torchvision for neural-network backbones and inference
- NumPy for numerical operations
- Pillow for stimulus transformations and rendering
- Matplotlib and pandas for plots and tabular summaries

Choose PyTorch and torchvision builds that match your CUDA installation. For a
CPU-only run, pass `--device cpu`.

## Project structure

```text
IVSN/
├── README.md
├── requirements.txt
├── images/
├── GIF/
└── codes/
    ├── ivsn_invariant_search.py   # backward-compatible CLI entry point
    ├── ivsn_invariance/
    │   ├── cli.py                 # arguments and experiment orchestration
    │   ├── runtime.py             # constants, seeds, and runtime geometry
    │   ├── domain.py              # trial and transformation data classes
    │   ├── data.py                # dataset and manifest I/O
    │   ├── imaging.py             # transformations and stimulus rendering
    │   ├── trials.py              # conditions and trial generation
    │   ├── models.py              # feature extractors and attention models
    │   ├── search.py              # IVSN fixation search
    │   ├── visualization.py       # attention and example figures
    │   └── reporting.py           # summaries, CSV files, and plots
    ├── model_weights/             # add local checkpoints here (not in Git)
    └── run_*.sh
```

## Model weights (not included)

Model checkpoints are intentionally excluded from Git because they are too large
for a normal GitHub repository. After cloning the repository, copy the available
weights into:

```text
codes/model_weights/
```

Use these exact filenames:

```text
codes/model_weights/vgg_gist_model_epoch_25.pth
codes/model_weights/conv_gist_model_epoch_15.pth
codes/model_weights/conv_gist_mlp_model_epoch_10.pth
codes/model_weights/vgg_gist_imagenet64_epoch25.pth
```

For example, on Linux or macOS:

```bash
cp /path/to/weights/*.pth codes/model_weights/
```

On Windows PowerShell:

```powershell
Copy-Item C:\path\to\weights\*.pth codes\model_weights\
```

The default paths are resolved relative to the source files, not the current
working directory. The program therefore finds the weights whether it is started
from the repository root or from `codes/`. A custom location can still be passed
through the corresponding `--*-checkpoint` option.

## Dataset

The invariance experiment expects one folder per category below `--data-root`.
Each category folder must contain PNG images. Supported configurations contain
either six or eight categories:

```text
sheep, cattle, cats, horses, teddybears, kites[, dogs, elephants]
```

The alias `teddy_bears` is also accepted for `teddybears`.

The dataset used by the original IVSN experiments is available
[here](https://drive.google.com/file/d/1ti0MT860zGEUu18BCCe9QEBHa46yBnC_/view?usp=drive_link).

## Running the invariance experiment

From the repository root:

```bash
python codes/ivsn_invariant_search.py \
  --data-root /path/to/dataset \
  --out-dir codes/outputs/rotation_vgg \
  --transform-mode rotation \
  --model-kind vgg \
  --device cuda
```

The package entry point is equivalent when run from `codes/`:

```bash
cd codes
python -m ivsn_invariance --help
```

Available model kinds are `vgg`, `vgg_gist_pretrained`, `conv_gist`,
`conv_gist_mlp`, and `vgg_gist_imagenet64`. Existing `run_*.sh` scripts provide
complete examples for these variants.

## Outputs

Each experiment writes trial manifests, per-trial JSON/CSV results, grouped
summaries, plots, and optional example visualizations below `--out-dir`.

## Original IVSN publication

The original IVSN model was published in *Nature Communications*:
[Finding any Waldo with zero-shot invariant and efficient visual search](https://www.nature.com/articles/s41467-018-06217-x).

```bibtex
@article{zhang2018finding,
  title={Finding any Waldo with zero-shot invariant and efficient visual search},
  author={Zhang, Mengmi and Feng, Jiashi and Ma, Keng Teck and Lim, Joo Hwee and Zhao, Qi and Kreiman, Gabriel},
  journal={Nature Communications},
  volume={9},
  number={1},
  pages={3730},
  year={2018},
  publisher={Nature Publishing Group UK London}
}
```

## License

Licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
Commercial use requires formal permission.
