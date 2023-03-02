const switchVisualization = () => {
    const MidiPlayer = document.getElementById('midi-player');
    const pianoRoll = document.getElementById('piano-roll-visualizer');
    const staff = document.getElementById('staff-visualizer');
    if (pianoRoll.style.visibility != 'hidden') {
        pianoRoll.style.visibility = 'hidden';
        staff.style.visibility = 'visible';
        MidiPlayer.setAttribute('visualizer', '#staff-visualizer');
    } else {
        staff.style.visibility = 'hidden';
        pianoRoll.style.visibility = 'visible';
        MidiPlayer.setAttribute('visualizer', '#piano-roll-visualizer');
    }
}
