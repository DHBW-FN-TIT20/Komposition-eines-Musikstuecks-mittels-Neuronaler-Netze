let abcString = '';
let xmlhttp = new XMLHttpRequest();
const abcFileLocation = document.getElementById('abc-file-location').getAttribute('abcfilelocation');
console.log(abcFileLocation);
xmlhttp.open('GET', abcFileLocation, false);
xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status == 200 && xmlhttp.readyState == 4){
        abcString = xmlhttp.responseText;
    }
};
xmlhttp.send();

abcString = abcString ? abcString : 'X:1\nK:D\nDD AA|BBA2|\n';
abcString = abcString.split(/\r?\n/) // Split input text into an array of lines
    .filter(line => line.trim() !== "") // Filter out lines that are empty or contain only whitespace
    .join("\n"); // Join line array into a string

let visualOptions = {  };
window.ABCJS.renderAbc('abcjs', abcString, visualOptions);
