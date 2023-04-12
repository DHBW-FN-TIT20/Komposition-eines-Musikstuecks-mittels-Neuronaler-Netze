const links = $('#links');
const infoButton = $('#toggleInfo');
const help = $('#help');
const helpButton = $('#toggleHelp');
const body = $('body');
const sideboardButton = $('#toggle-sideboard');

let abcString = '';
let xmlhttp = new XMLHttpRequest();
const abcFileLocation = $('#abc-file-location').attr('abcfilelocation');
xmlhttp.open('GET', abcFileLocation, false);
xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status == 200 && xmlhttp.readyState == 4){
        abcString = xmlhttp.responseText;
    }
};
xmlhttp.send();

abcString = abcString ? abcString : 'X:1\nT:Keine passende ABC-Datei gefunden';
abcString = abcString.split(/\r?\n/) // Split input text into an array of lines
    .filter(line => line.trim() !== '') // Filter out lines that are empty or contain only whitespace
    .join('\n'); // Join line array into a string

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
ABCJS.renderAbc('abcjs', abcString, visualOptions);

let toggleInfo = () => {
    if (links.attr('isHidden') == 'True') {
        links.attr('isHidden', 'False');
        infoButton.html('x');
        help.attr('isHidden', 'True');
        helpButton.html('?');
    } else {
        links.attr('isHidden', 'True');
        infoButton.html('i');
    }
}

let toggleHelp = () => {
    if (help.attr('isHidden') == 'True') {
        help.attr('isHidden', 'False');
        helpButton.html('x');
        links.attr('isHidden', 'True');
        infoButton.html('i');
    } else {
        help.attr('isHidden', 'True');
        helpButton.html('?');
    }
}

let toggleSideboard = () => {
    console.log(body.css('grid-template-columns'));
    if (body.css('grid-template-columns').startsWith('0px')) {
        body.css('grid-template-columns', 'var(--sideboard-width) 5px calc(100% - 5px - var(--sideboard-width))');
        sideboardButton.html('&#706; close');
    } else {
        body.css('grid-template-columns', '0 5px calc(100% - 5px)');
        sideboardButton.html('open &#707;');
    }
}

let generate_new_song = () => {
    const main = $('#main');
    const loadingScreen = $('#loading-screen');
    const generateButton = $('#generate-button');
    $('#generate input[type=radio]').each(function() {
        $(this).prop('disabled', true);
    });

    main.scrollTop = 0;
    main.css('overflow', 'hidden');
    loadingScreen.css('display', 'flex');
    generateButton.prop('disabled', true);

    let params = {
        model: $(`input[type='radio'][name='model']:checked`).val(),
        length: $(`input[type='radio'][name='length']:checked`).val(),
        music: $(`input[type='radio'][name='music']:checked`).val(),
        coding: $(`input[type='radio'][name='coding']:checked`).val(),
        instrument: $(`input[type='radio'][name='instrument']:checked`).val(),
        bpm: $(`input[type='radio'][name='bpm']:checked`).val()
    };

    let url = '/';
    let data = '';
    $.get({
        url: '/generate',
        data: params,
        success: d => {
            data = d;
        },
        dataType: 'text'
    }).done(() => {
        console.log(`Data from /generate: ${data}`);
        url += data;
    }).fail(() => {
        url += '';
        alert('Leider ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut oder wenden Sie sich an den Webseiten-Administrator.');
    }).always(() => {
        document.location.href = url;
    });
}
