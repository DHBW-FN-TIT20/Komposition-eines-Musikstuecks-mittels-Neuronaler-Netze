from os.path import dirname

import music21 as m21
from mukkeBude import utils as mukkeBude_utils
from mukkeBude.mapping import MusicMapping
from mukkeBude.model import MukkeBudeLSTM
from mukkeBude.model import MukkeBudeTransformer

midiLocation = dirname(__file__) + "/static/midi"
mxlLocation = dirname(__file__) + "/static/mxl"

mapping = MusicMapping.create()
models = {
    "LSTM": {
        "Bach": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="Bach_soloMelodie_lstm"),
            "Polyphonie": MukkeBudeLSTM.load(mapping=mapping, name="Bach_polyphonie_lstm"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "PinkFloyd": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="PinkFloyd_soloMelodie_lstm"),
            "Polyphonie": MukkeBudeLSTM.load(mapping=mapping, name="PinkFloyd_polyphonie_lstm"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "Videospielmusik": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="Videospielmusik_soloMelodie_lstm"),
            "Polyphonie": MukkeBudeLSTM.load(mapping=mapping, name="Videospielmusik_polyphonie_lstm"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
    },
    "Transformer": {
        "Bach": {
            "SoloMelodie": MukkeBudeTransformer.load(mapping=mapping, name="Bach_soloMelodie_transformer"),
            "Polyphonie": MukkeBudeTransformer.load(mapping=mapping, name="Bach_polyphonie_transformer"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "PinkFloyd": {
            "SoloMelodie": MukkeBudeTransformer.load(mapping=mapping, name="PinkFloyd_soloMelodie_transformer"),
            "Polyphonie": MukkeBudeTransformer.load(mapping=mapping, name="PinkFloyd_polyphonie_transformer"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "Videospielmusik": {
            "SoloMelodie": MukkeBudeTransformer.load(mapping=mapping, name="Videospielmusik_soloMelodie_transformer"),
            "Polyphonie": MukkeBudeTransformer.load(mapping=mapping, name="Videospielmusik_polyphonie_transformer"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
    },
}

m21Instrument = {
    "Piano": m21.instrument.Piano(),
    "Gitarre": m21.instrument.Guitar(),
}

settings = [
    {
        "id": "model",
        "name": "Model",
        "radioButtons": [
            {
                "id": "lstm",
                "value": "LSTM",
                "label": "LSTM",
                "checked": True,
            },
            {
                "id": "transformer",
                "value": "Transformer",
                "label": "Transformer",
                "checked": False,
            },
        ],
    },
    {
        "id": "length",
        "name": "Länge",
        "radioButtons": [
            {
                "id": "lgth1",
                "value": "5",
                "label": "max. 5 Takte",
                "checked": True,
            },
            {
                "id": "lgth2",
                "value": "10",
                "label": "max. 10 Takte",
                "checked": False,
            },
            {
                "id": "lgth3",
                "value": "20",
                "label": "max. 20 Takte",
                "checked": False,
            },
        ],
    },
    {
        "id": "music",
        "name": "Musikrichtung",
        "radioButtons": [
            {
                "id": "bach",
                "value": "Bach",
                "label": "Bach",
                "checked": True,
            },
            {
                "id": "pink-floyd",
                "value": "PinkFloyd",
                "label": "Pink Floyd",
                "checked": False,
            },
            {
                "id": "videogames",
                "value": "Videospielmusik",
                "label": "Videospielmusik",
                "checked": False,
            },
        ],
    },
    {
        "id": "coding",
        "name": "Codierung",
        "radioButtons": [
            {
                "id": "melody",
                "value": "SoloMelodie",
                "label": "Solo-Melodie",
                "checked": True,
            },
            {
                "id": "polyphon",
                "value": "Polyphon",
                "label": "Polyphon",
                "checked": False,
            },
        ],
    },
    {
        "id": "instrument",
        "name": "Instrument",
        "radioButtons": [
            {
                "id": "piano",
                "value": "Piano",
                "label": "Piano",
                "checked": True,
            },
            {
                "id": "guitar",
                "value": "Gitarre",
                "label": "Gitarre",
                "checked": False,
            },
        ],
    },
    {
        "id": "bpm",
        "name": "BPM",
        "radioButtons": [
            {
                "id": "90",
                "value": "90",
                "label": "90",
                "checked": True,
            },
            {
                "id": "120",
                "value": "120",
                "label": "120",
                "checked": False,
            },
        ],
    },
]
