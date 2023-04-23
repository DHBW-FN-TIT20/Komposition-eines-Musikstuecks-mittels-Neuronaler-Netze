from typing import Union
import time
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from utils import *

app = Flask(__name__)

@app.route("/")
@app.route("/<midi>")
def midi_viewer(midi=None):
    return render_template(
        "index.html",
        midi=midi,
        placeholder="generated_song_p0_8_withSeed",
        midiExample=False,
        settings=settings,
    )


@app.route("/examples")
@app.route("/examples/<midiExample>")
def midi_viewer_example(midiExample=None):
    return render_template(
        "index.html",
        midi=None,
        placeholder="generated_song_p0_8_withSeed",
        midiExample=midiExample,
        settings=settings,
    )


@app.route("/generate")
def return_generated_name():
    model = request.args.get("model") or "LSTM"
    length = request.args.get("length") or 5
    music = request.args.get("music") or "bach"
    coding = request.args.get("coding") or "SoloMelodie"
    instrument = request.args.get("instrument") or "Piano"
    bpm = request.args.get("bpm") or 90

    timestamp = str(time.time())
    timestamp = timestamp.replace('.', '')
    generatedName = model + length + music + coding + instrument + bpm + timestamp

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

    mukkeBude_utils.write_midi(new_song, f"{midiLocation}/{generatedName}.mid")
    mukkeBude_utils.write_musicxml(new_song, f"{mxlLocation}/{generatedName}.musicxml")

    return generatedName
