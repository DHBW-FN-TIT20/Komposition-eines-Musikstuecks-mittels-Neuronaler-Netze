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
