# Komposition-eines-Musikstuecks-mittels-Neuronaler-Netze
Ziel der Studienarbeit ist die Komposition eines kleinen Musikstücks mithilfe eines Neuronalen Netzes.

![](./flask-webapp/static/img/mukkebude-readme.png)

Es stehen LSTM, GRU und Transformer-Netze zur Verfügung. Die Netze werden mit Hilfe von Keras und Tensorflow implementiert. Es sind bereits trainierte Modelle vorhanden, die in der Webapp oder auch in den Jupyter-Notebook's ([demos](./demos/)) verwendet werden können.

# Usage
Hier wird beschrieben wir man das Projekt verwendet.

**Achtung!** </br>
Die Verwendung der Transformer-Netze ist nicht nativ über die reine Installation des Projektes über PyPi verwendbar. Hierfür muss das Porjekt mittels `git clone` heruntergeladen werden.

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

Für die Verwendung der **Jupyter-Notebooks** muss jupyter-lab zusätzlich installiert werden!
```bash
pip install jupyterlab
```

## Verwendung
Nach einer erfolgreichen installtion kann das modul mittels `import mukkeBude` verwendet werden. Entsprechende Beispiele sind unter [demos](./demos/) zu finden.

# Webapp
Im Ordner [flask-webapp](./flask-webapp/) befindet sich eine Webapp, die die Verwendung der Netze ermöglicht. Für das Starten der Webapp sollte mukkeBude entsprechend installiert werden. Zusätzliche Abhängigkeiten können mittels `pip install -r requirements-dev.txt` im Ordner [flask-webapp](./flask-webapp/) installiert werden.

Starten der Webapp:
```bash
cd flask-webapp
flask run
```

Starten mit Docker:
```bash
# Bauen des Docker-Images
docker buildx build -t mukkebude .

# Starten des Docker-Containers
docker run -p 8080:8080 -d --name mukkebude mukkebude
```

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
