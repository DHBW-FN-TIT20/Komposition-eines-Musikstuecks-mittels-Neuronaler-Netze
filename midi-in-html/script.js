const switchVisualization = () => {
    const midiPlayer = document.getElementById('midi-player');
    const midiPlayer1 = document.getElementById('midi-player1');
    const midiPlayer2 = document.getElementById('midi-player2');
    const visButton = document.getElementById('vis-button');
    if (midiPlayer.getAttribute('visualizer') == '#piano-roll-visualizer') {
        midiPlayer.setAttribute('visualizer', '#staff-visualizer');
        midiPlayer1.setAttribute('visualizer', '#staff-visualizer1');
        midiPlayer2.setAttribute('visualizer', '#staff-visualizer2');
        visButton.innerHTML = 'Follow on: Sheet Music';
    } else {
        midiPlayer.setAttribute('visualizer', '#piano-roll-visualizer');
        midiPlayer1.setAttribute('visualizer', '#piano-roll-visualizer1');
        midiPlayer2.setAttribute('visualizer', '#piano-roll-visualizer2');
        visButton.innerHTML = 'Follow on: Piano Roll';
    }
}
