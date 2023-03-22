const links = document.getElementById('links');
const infoButton = document.getElementById('toggleInfo');
const help = document.getElementById('help');
const helpButton = document.getElementById('toggleHelp');

let abcString = '';
let xmlhttp = new XMLHttpRequest();
const abcFileLocation = document.getElementById('abc-file-location').getAttribute('abcfilelocation');
xmlhttp.open('GET', abcFileLocation, false);
xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status == 200 && xmlhttp.readyState == 4){
        abcString = xmlhttp.responseText;
    }
};
xmlhttp.send();

abcString = abcString ? abcString : 'X:1\nT:Keine passende ABC-Datei gefunden';
abcString = abcString.split(/\r?\n/) // Split input text into an array of lines
    .filter(line => line.trim() !== "") // Filter out lines that are empty or contain only whitespace
    .join("\n"); // Join line array into a string

let visualOptions = {
    format: {
        titlefont: 'Inconsolata bold 24',
        subtitlefont: 'Inconsolata bold',
        composerfont: 'Inconsolata bold',
        partsfont: 'Inconsolata bold',
        tempofont: 'Inconsolata bold',
        gchordfont: 'Inconsolata bold',
        annotationfont: 'Inconsolata bold',
        infofont: 'Inconsolata bold',
        textfont: 'Inconsolata bold',
        vocalfont: 'Inconsolata bold',
        wordsfont: 'Inconsolata bold'
    },
    // responsive: 'resize',
    staffwidth: 740,
    wrap: {
        minSpacing: 1.5,
        maxSpacing: 2.5,
        preferredMeasuresPerLine: 4,
    }
};
window.ABCJS.renderAbc('abcjs', abcString, visualOptions);

let toggleInfo = () => {
    if (links.getAttribute('isHidden') == 'True') {
        links.setAttribute('isHidden', 'False');
        infoButton.innerHTML = 'x';
        help.setAttribute('isHidden', 'True');
        helpButton.innerHTML = '?';
    } else {
        links.setAttribute('isHidden', 'True');
        infoButton.innerHTML = 'i';
    }
}

let toggleHelp = () => {
    if (help.getAttribute('isHidden') == 'True') {
        help.setAttribute('isHidden', 'False');
        helpButton.innerHTML = 'x';
        links.setAttribute('isHidden', 'True');
        infoButton.innerHTML = 'i';
    } else {
        help.setAttribute('isHidden', 'True');
        helpButton.innerHTML = '?';
    }
}

let generate_new_song = () => {
    document.getElementById('generate-button').disabled = true;
    let params = {
        model: $("input[type='radio'][name='model']:checked").val(),
        length: $("input[type='radio'][name='length']:checked").val(),
        music: $("input[type='radio'][name='music']:checked").val(),
        coding: $("input[type='radio'][name='coding']:checked").val(),
        instrument: $("input[type='radio'][name='instrument']:checked").val(),
        bpm: $("input[type='radio'][name='bpm']:checked").val()
    };

    let url = '/';
    let data = '';
    $.get({
        url: '/generate',
        data: params,
        success: d => {
            data = d;
        },
        dataType: "text"
    }).done(() => {
        console.log(`Data from /generate: ${data}`);
        url += data;
    }).fail(() => {
        url += '';
        // alert('Leider ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut oder wenden Sie sich an den Webseiten-Administrator.');
    }).always(() => {
        document.location.href = url;
    });
}
