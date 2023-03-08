from flask import Flask, render_template
from os import listdir
from os.path import isfile, join

midiExamples = [f for f in listdir('static/midi-examples') if isfile(join('static/midi-examples', f))]
abcExamples = [f for f in listdir('static/abc-examples') if isfile(join('static/abc-examples', f))]

exampleFiles = [f for f in midiExamples if f'{f}.abc' in abcExamples]

app = Flask(__name__)

@app.route('/')
@app.route('/<midi>')
def midi_viewer(midi=None, midiExamples=midiExamples):
    return render_template('index.html', midi=midi, exampleFiles=exampleFiles)
