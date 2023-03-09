from flask import Flask, render_template
from utils import *

app = Flask(__name__)

@app.route('/')
@app.route('/<midi>')
def midi_viewer(midi=None):
    return render_template('index.html', midi=midi, settings=settings, exampleFiles=exampleFiles)
