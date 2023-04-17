from os.path import dirname

midiLocation = dirname(__file__) + "/static/midi"
abcLocation = dirname(__file__) + "/static/abc"

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
