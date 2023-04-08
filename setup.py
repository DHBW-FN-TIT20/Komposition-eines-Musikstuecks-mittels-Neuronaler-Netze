from setuptools import setup
from setuptools import find_packages

setup(
    name='mukkeBude',
    version='0.0.1',
    description='A music generation library with transformer and lstm models',
    readme='README.md',
    author="Florian Glaser, Florian Herkommer, David Felder",
    author_email="Florian.Glaser@ifm.com, Florian.Herkommer@ifm.com, David.Felder@ifm.com",
    url="https://github.com/DHBW-FN-TIT20/Komposition-eines-Musikstuecks-mittels-Neuronaler-Netze/",
    install_requires=[
        'importlib-metadata; python_version == "3.9"',
        'setuptools',
        'wheel',
        'tensorflow == 2.10.1',
        'cuda-python',
        'nvidia-cudnn',
        'nvidia-cuda-nvcc',
        'music21',
        'keras-nlp',
    ],
    packages=find_packages(
        include="mukkeBude",
    ),
)