from os.path import dirname

import music21 as m21

from mukkeBude.mapping import MusicMapping
from mukkeBude.model import MukkeBudeLSTM

midiLocation = dirname(__file__) + "/static/midi"
mxlLocation = dirname(__file__) + "/static/mxl"

mapping = MusicMapping.create()
models = {
    "LSTM": {
        "Bach": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="Bach_soloMelodie_lstm"),
            "Polyphon": MukkeBudeLSTM.load(mapping=mapping, name="Bach_polyphonie_lstm"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphon": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "PinkFloyd": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="PinkFloyd_soloMelodie_lstm"),
            "Polyphon": MukkeBudeLSTM.load(mapping=mapping, name="PinkFloyd_polyphonie_lstm"),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphon": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "Videospielmusik": {
            "SoloMelodie": MukkeBudeLSTM.load(mapping=mapping, name="Videospielmusik_soloMelodie_lstm"),
            "Polyphon": MukkeBudeLSTM.load(mapping=mapping, name="Videospielmusik_polyphonie_lstm"),
            "SeedSoloMelodie": "n79 _ n67 n67 n67 _ n74 n79 n77 _ n65 n65 n65 _ n72 n77 n76 _ n60 n60 n60 _ n76 n76 n76 _ _ n72 n67 n76 n78 _ _ _ n74 n69 n81 n67 _ _ _ _ _ _ _ _ _ _",
            "SeedPolyphon": "n46 d4 xxsep d4 n53 d2 n50 d2 xxsep d2 n62 d1 n58 d1 xxsep d2 n53 d2 n53 d2 n50 d2 xxsep d4 n46 d4 xxsep d4 n55 d2 n51 d2 xxsep d2 n63 d1 n58 d1 xxsep d2 n55 d2 n55 d2 n51 d2 xxsep d4 n46 d4 xxsep d4 n57 d2 n53 d2 xxsep d2 n65 d1 n60 d1 xxsep d2 n57 d2 n57 d2 n53 d2 xxsep d4 n46 d4 xxsep d4 n55 d2 n51 d2 xxsep d2 n63 d1 n58 d1 xxsep d2 n55 d2 n55 d2 n51 d2 xxsep d4",
        },
    },
    # Currently not working, only uncomment if Transformer network is working
    # "Transformer": {
    #     "Bach": {
    #         "SoloMelodie": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="Bach_soloMelodie_transformer",
    #             path="../demos/raw_train_ds_mono_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "Polyphon": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="Bach_polyphonie_transformer",
    #             path="../demos/raw_train_ds_poly_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
    #         "SeedPolyphon": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
    #     },
    #     "PinkFloyd": {
    #         "SoloMelodie": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="PinkFloyd_soloMelodie_transformer",
    #             path="../demos/raw_train_ds_poly_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "Polyphon": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="PinkFloyd_polyphonie_transformer",
    #             path="../demos/raw_train_ds_poly_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
    #         "SeedPolyphon": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
    #     },
    #     "Videospielmusik": {
    #         "SoloMelodie": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="Videospielmusik_soloMelodie_transformer",
    #             path="../demos/raw_train_ds_poly_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "Polyphon": MukkeBudeTransformer.load(
    #             mapping=mapping,
    #             name="Videospielmusik_polyphonie_transformer",
    #             path="../demos/raw_train_ds_poly_bach.txt",
    #             min_training_seq_len=32,
    #         ),
    #         "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
    #         "SeedPolyphon": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
    #     },
    # },
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
            # Currently not working, only uncomment if Transformer network is working
            # {
            #     "id": "transformer",
            #     "value": "Transformer",
            #     "label": "Transformer",
            #     "checked": False,
            # },
        ],
    },
    {
        "id": "length",
        "name": "Zeichenlänge",
        "radioButtons": [
            {
                "id": "lgth1",
                "value": "200",
                "label": "max. 200 Zeichen",
                "checked": True,
            },
            {
                "id": "lgth2",
                "value": "500",
                "label": "max. 500 Zeichen",
                "checked": False,
            },
            {
                "id": "lgth3",
                "value": "1000",
                "label": "max. 1000 Zeichen",
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
    {
        "id": "key",
        "name": "Tonart",
        "radioButtons": [
            {
                "id": "C",
                "value": "C",
                "label": "C-Dur / a-Moll",
                "checked": True,
            },
            {
                "id": "E",
                "value": "E",
                "label": "E-Dur / cis-Moll",
                "checked": False,
            },
            {
                "id": "G",
                "value": "G",
                "label": "G-Dur / e-Moll",
                "checked": False,
            },
        ],
    },
]
