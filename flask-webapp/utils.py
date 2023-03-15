from os import listdir
from os.path import isfile, join, splitext

midiLocation = 'static/midi/examples'
abcLocation = 'static/abc/examples'
midiExamples = [f for f in listdir(midiLocation) if isfile(join(midiLocation, f))]
abcExamples = [f for f in listdir(abcLocation) if isfile(join(abcLocation, f))]
exampleFiles = [splitext(f)[0] for f in midiExamples if f'{splitext(f)[0]}.abc' in abcExamples]

settings = [
    {
        'id': 'model',
        'name': 'Model',
        'radioButtons': [
            {
                'id': 'lstm',
                'value': 'LSTM',
                'label': 'LSTM',
                'checked': True
            },
            {
                'id': 'transformer',
                'value': 'Transformer',
                'label': 'Transformer',
                'checked': False
            }
        ]
    },
    {
        'id': 'length',
        'name': 'Länge',
        'radioButtons': [
            {
                'id': 'lgth1',
                'value': '5',
                'label': '5 Takte',
                'checked': True
            },
            {
                'id': 'lgth2',
                'value': '10',
                'label': '10 Takte',
                'checked': False
            },
            {
                'id': 'lgth3',
                'value': '20',
                'label': '20 Takte',
                'checked': False
            }
        ]
    },
    {
        'id': 'music',
        'name': 'Musikrichtung',
        'radioButtons': [
            {
                'id': 'bach',
                'value': 'Bach',
                'label': 'Bach',
                'checked': True
            },
            {
                'id': 'pink-floyd',
                'value': 'PinkFloyd',
                'label': 'Pink Floyd',
                'checked': False
            },
            {
                'id': 'videogames',
                'value': 'Videospielmusik',
                'label': 'Videospielmusik',
                'checked': False
            }
        ]
    },
    {
        'id': 'coding',
        'name': 'Codierung',
        'radioButtons': [
            {
                'id': 'melody',
                'value': 'SoloMelodie',
                'label': 'Solo-Melodie',
                'checked': True
            },
            {
                'id': 'polyphon',
                'value': 'Polyphon',
                'label': 'Polyphon',
                'checked': False
            }
        ]
    }
]
