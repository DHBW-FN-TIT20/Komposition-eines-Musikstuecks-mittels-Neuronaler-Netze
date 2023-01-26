import os
import json
import music21 as m21
import keras
import numpy as np

KERN_DATASET_PATH = "mukkeBude\\deutschl\\erk"
SAVE_DIR = "mukkeBude\\dataset"
SINGLE_FILE_DATASET = "mukkeBude\\file_dataset"
MAPPING_PATH = "mukkeBude\\mapping.json"
SEQUENCE_LENGTH = 64 # ca. 4 Takte im 4/4 Takt
ACCEPTABLE_DURATIONS = [
    0.25, #sixteenth note
    0.5, #eight note
    0.75, #dotted eight note
    1, #quarter note
    1.5, #dotted quarter note
    2, #half note
    3, # dotted half note
    4 # full note
]
# kern, MIDI, MusicXML, ABC -> m21 -> kern, MIDI, ...

def has_acceptable_durations(song, acceptable_durations):
    # filter notes and rests from song first
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def load_songs_in_kern(dataset_path):
    songs = []
    # go through all the files in dataset and load the with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] ==  "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def encode_song(song : m21.stream.Score, time_step=0.25):
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]
    encoded_song = []
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # e.g. = 60
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        #convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def transpose(song : m21.stream.Score):
    """Transpose every song in Cmajor (CDur) or Aminor (AMoll) so the model does not have to learn all 24 keys (Tonarten). 
    If we want that we have to transpose the songs to all 24 keys.

    Args:
        song (m21.stream.Score): _description_

    Returns:
        m21.stream.Score: transposed song
    """
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    print(key)

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major": #Dur?
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor": #Moll?
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
         
    # transpose song by calculated interval with music21
    transposed_song = song.transpose(interval)
    return transposed_song

def preprocess(dataset_path):
    # load the songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):        
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin (C-Dur/A-Moll)
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i) )
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]
    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    return mappings

def convert_songs_to_int(songs, mapping_path):
    int_songs = []
    # load mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length, file_dataset_path, mapping_path):
    # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13; i: [12, 13], t: 14
    # load songs and map them to int
    songs = load(file_dataset_path)
    int_songs = convert_songs_to_int(songs, mapping_path)

    # generate the training sequences
    # e.g.  100 symbols, 64 sequence_length, 100-64 = 36
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    # inputs: (# of sequences, sequence length, vocabulary size)
    # [ [0, 1, 2], [1, 1, 2] ] ->  [ [ [1, 0, 0 ], [0, 1, 0], [0, 0, 1] ], []]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    mappings = create_mapping(songs, MAPPING_PATH)


if __name__ == "__main__":
    main()