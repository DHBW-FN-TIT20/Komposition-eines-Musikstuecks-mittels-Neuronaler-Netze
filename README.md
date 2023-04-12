# Komposition-eines-Musikstuecks-mittels-Neuronaler-Netze
Ziel der Studienarbeit ist die Komposition eines kleinen Musikstücks. Die Komposition erfolgt mittels eines Neuronalen Netzes.

![](./flask-webapp/static/img/mukkebude.png)

# Usage
Hier wird beschrieben wir man das Projekt verwendet.

## Installation
Um die unter [demos](./demos/) bereitgestellten jupyter-notebook verwenden zu können, muss das Projekt mittels pip installiert werden.
Hierfür gibt es folgende Möglichkeiten:

**GitHub Repo**
```bash
# Clone das Repo
git clone git@github.com:DHBW-FN-TIT20/Komposition-eines-Musikstuecks-mittels-Neuronaler-Netze.git mukkeBude
cd mukkeBude

# Installieren mittels pip
pip install .
```

**PyPi**
```bash
pip install mukkeBude
```

Für die Verwendung der **Jupyter-Notebooks** muss jupyter-lab zusätzlich installiert werden!
```bash
pip install jupyterlab
```

## Verwendung
Nach einer erfolgreichen installtion kann das modul mittels `import mukkeBude` verwendet werden. Entsprechende Beispiele sind unter [demos](./demos/) zu finden.

# Developing

Hier wird beschrieben, wie man seine Entwicklungsumgebung entsprechend vorbereitet, um an dem Projekt zu entwicklen.
Empfohlen ist die Verwendung von **Conda**, da hier die Verwendung von der GPU deutlich einfach ist. Bei der Verwendung der anderen Methoden müssen
unter Umständen weitere Schritte unternommen werden, um die GPU zu verwenden.

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
pip install -r requirements-dev.txt
```

Mit venv:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt
```

## Poetry
```bash
poetry install --with=dev
poetry shell
```
