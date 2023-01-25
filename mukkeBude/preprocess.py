import os
import music21 as m21

# kern, MIDI, MusicXML, ABC -> m21 -> kern, MIDI, ...
KERN_DATASET_PATH = "mukkeBude\deutschl"

def load_songs_in_kern(dataset_path):
    songs = []
    # go through all the files in dataset and load the with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] ==  "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def preprocess(dataset_path):
    pass

    # load the songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    # filter out songs that have non-acceptable durations

    # transpose songs to Cmaj/Amin (C-Dur/A-Moll)

    # encode songs with music time series representation

    # save songs to text file


if __name__ == "__main__":
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    song.show()