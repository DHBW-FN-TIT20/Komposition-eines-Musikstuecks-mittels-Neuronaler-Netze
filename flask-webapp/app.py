from flask import Flask, render_template, request
from utils import *

app = Flask(__name__)

@app.route('/')
@app.route('/<midi>')
def midi_viewer(midi=None):
    return render_template('index.html', midi=midi, placeholder='Zelda', midiExample=False, settings=settings, exampleFiles=exampleFiles)

@app.route('/examples')
@app.route('/examples/<midiExample>')
def midi_viewer_example(midiExample=None):
    return render_template('index.html', midi=None, placeholder='Zelda', midiExample=midiExample, settings=settings, exampleFiles=exampleFiles)

@app.route('/generate')
def return_generated_name():
    model = request.args.get('model') or 'LSTM'
    length = request.args.get('length') or 5
    music = request.args.get('music') or 'Bach'
    coding = request.args.get('coding') or 'SoloMelodie'
    instrument = request.args.get('instrument') or 'Piano'
    bpm = request.args.get('bpm') or 90

    # TODO generation of song with params from GUI
    generatedName = model + length + music + coding + instrument + bpm
    return generatedName
