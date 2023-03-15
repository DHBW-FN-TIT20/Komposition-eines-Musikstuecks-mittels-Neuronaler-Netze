from flask import Flask, render_template
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
