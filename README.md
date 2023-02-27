# Komposition-eines-Musikstuecks-mittels-Neuronaler-Netze
Ziel der Studienarbeit ist die Komposition eines kleinen Musikstücks. Die Komposition erfolgt mittels eines Neuronalen Netzes.

# Setup

## Conda
Installation mithilfe von conda:

```bash
conda env create -f environment.yml
conda activate tf-gpu

# Enable GPU support on Linux (need to be done in every new shell)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

## Pip
Ohne venv:

```bash
pip install -r requirements.txt

# or for dev
pip install -r requirements-dev.txt
```

Mit venv:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# or for dev
pip install -r requirements-dev.txt
```