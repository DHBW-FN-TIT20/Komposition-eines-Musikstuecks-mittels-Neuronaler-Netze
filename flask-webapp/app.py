# Disable tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

from typing import Union

import music21 as m21
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from utils import *

from mukkeBude import utils as mukkeBude_utils
from mukkeBude.mapping import MusicMapping
from mukkeBude.model import MukkeBudeLSTM
from mukkeBude.model import MukkeBudeTransformer

app = Flask(__name__)
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
            "SoloMelodie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="Bach_soloMelodie_transformer",
                path="raw_train_ds_mono_bach.txt",
                min_training_seq_len=32,
            ),
            "Polyphonie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="Bach_polyphonie_transformer",
                path="raw_train_ds_poly_bach.txt",
                min_training_seq_len=32,
            ),
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "PinkFloyd": {
            "SoloMelodie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="PinkFloyd_soloMelodie_transformer",
                path="raw_train_ds_mono_bach.txt",
                min_training_seq_len=32,
            ),  # TODO: change path
            "Polyphonie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="PinkFloyd_polyphonie_transformer",
                path="raw_train_ds_mono_bach.txt",
                min_training_seq_len=32,
            ),  # TODO: change path
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
        "Videospielmusik": {
            "SoloMelodie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="Videospielmusik_soloMelodie_transformer",
                path="raw_train_ds_mono_bach.txt",
                min_training_seq_len=32,
            ),  # TODO: change path
            "Polyphonie": MukkeBudeTransformer.load(
                mapping=mapping,
                name="Videospielmusik_polyphonie_transformer",
                path="raw_train_ds_mono_bach.txt",
                min_training_seq_len=32,
            ),  # TODO: change path
            "SeedSoloMelodie": "n72 _ _ _ _ _ n72 _ _ _ _ _ n72 _ n71 _",
            "SeedPolyphonie": "xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep",
        },
    },
}

m21Instrument = {
    "Piano": m21.instrument.Piano(),
    "Gitarre": m21.instrument.Guitar(),
}


@app.route("/")
@app.route("/<midi>")
def midi_viewer(midi=None):
    return render_template(
        "index.html",
        midi=midi,
        placeholder="Zelda",
        midiExample=False,
        settings=settings,
        exampleFiles=exampleFiles,
    )


@app.route("/examples")
@app.route("/examples/<midiExample>")
def midi_viewer_example(midiExample=None):
    return render_template(
        "index.html",
        midi=None,
        placeholder="Zelda",
        midiExample=midiExample,
        settings=settings,
        exampleFiles=exampleFiles,
    )


@app.route("/generate")
def return_generated_name():
    model = request.args.get("model") or "LSTM"
    length = request.args.get("length") or 5
    music = request.args.get("music") or "bach"
    coding = request.args.get("coding") or "SoloMelodie"
    instrument = request.args.get("instrument") or "Piano"
    bpm = request.args.get("bpm") or 90

    generatedName = model + length + music + coding + instrument + bpm

    start_seed = models[model][music][f"Seed{coding}"]
    model: Union[MukkeBudeLSTM, MukkeBudeTransformer] = models[model][music][coding]
    generated_music = model.generate(start_seed=start_seed, max_length=int(length))

    new_song_ints = mapping.numericalize(generated_music.split(" "))
    new_song = mukkeBude_utils.from_polyphonic_encoding(
        index_arr=np.array(new_song_ints),
        mapping=mapping,
        bpm=int(bpm),
        instrument=m21Instrument[instrument],
    )

    mukkeBude_utils.write_midi(new_song, f"{midiLocation}/{generatedName}.midi")

    return generatedName
