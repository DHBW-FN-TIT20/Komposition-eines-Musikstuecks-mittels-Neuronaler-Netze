import json
import os
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import music21 as m21
import numpy as np

BPB = 4  # beats per bar
TIMESIG = f"{BPB}/4"  # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1  # separator value for numpy encoding
VALTCONT = -2  # numpy value for TCONT - needed for compressing chord array
SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10 * BPB * SAMPLE_FREQ) + 1  # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = 8 * BPB * SAMPLE_FREQ
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

SEQType = Enum("SEQType", "Mask, Sentence, Melody, Chords, Empty")


def create_train_data_json(encoded_song: np.ndarray, sequence_length=32) -> List[Dict]:
    num_sequences = len(encoded_song) - sequence_length

    data = []

    for i in range(num_sequences):
        input = str(encoded_song[i : i + sequence_length])
        target = str(encoded_song[i + sequence_length])

        data.append({"input": input, "labels": target})

    path = Path("data.json")
    path.unlink(missing_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=None)

    return data


def create_train_data(encoded_songs: List[str], path: os.PathLike) -> None:
    """Create training data from encoded songs. Each song is written to a new line.
    xxbos and xxpad tokens are removed.

    Args:
        encoded_songs (List[str]): list of encoded songs
        path (os.PathLike): path to file
    """
    # Check if file exists, if so, delete it
    if os.path.exists(path):
        os.remove(path)

    with open(path, "a") as f:
        for song in encoded_songs:
            song = song.replace("xxbos ", "").replace("xxpad ", "")
            f.write(song + "\n")


def read_single(file_path: os.PathLike) -> Union[m21.stream.Score, m21.stream.Part, m21.stream.Opus]:
    """Convert file of a song to music21.stream.Score. Accepted file types are .mid, .krn, .abc, .mxl, .musicxml

    Args:
        file_path (str): the path to file

    Returns:
        m21.stream.Score: the converted song
    """
    return m21.converter.parse(file_path)  # type: ignore


def write_midi(song: m21.stream.Score, output_path: str = "test.mid") -> None:
    """Export music21.stream.Score to midi format.

    Args:
        song (m21.stream.Score): the song
        output_path (str, optional): filename with path. Defaults to "test.mid".
    """
    song.write("midi", fp=output_path)


def write_musicxml(song: m21.stream.Score, output_path: str = "test.musicxml") -> None:
    """Export music21.stream.Score to musicxml format

    Args:
        song (m21.stream.Score): the song
        output_path (str, optional): filename with path. Defaults to "test.musicxml".
    """
    song.write("musicxml", fp=output_path)


def read_single_from_corpus(corpus_path: str) -> m21.stream.Score:
    """Convert file in music21 corpus to music21.stream.Score. Paths to corpus files can be retrieved using e.g. paths = music21.corpus.getComposer('bach')

    Args:
        corpus_path (str): the path to the file in corpus

    Returns:
        m21.stream.Score: the converted song
    """
    return m21.corpus.parse(corpus_path)


def read_all(files: List[os.PathLike]) -> List[Union[m21.stream.Score, m21.stream.Part, m21.stream.Opus]]:
    """Converts all files to List[music21.stream.Score]. Accepted file types are .mid, .midi, .krn, .abc, .mxl, .musicxml

    Args:
        files (os.PathLike): path to files

    Returns:
        List[m21.stream.Score]: list of converted songs
    """
    songs = []
    for file in files:
        song = read_single(file)
        songs.append(song)
    return songs


def transpose_songs(songs: List[m21.stream.Score]) -> List[m21.stream.Score]:
    """Transpose songs to c major or a minor

    :param songs: list of songs
    :return: list of transposed songs
    """
    transposed_songs = []

    for song in songs:
        parts = song.getElementsByClass(m21.stream.Part)
        measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
        try:
            key = measures_part0[0][4]  # There is the key signature of the most songs in THIS dataset
        except IndexError:
            key = None

        # If key siganture is found, we use music21's key guessing algorithm
        if not isinstance(key, m21.key.Key):
            key = song.analyze("key")

        # get interval for transposition
        if key.mode == "major":
            # Interval between tonic and C
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        elif key.mode == "minor":
            # Interval between tonic and A
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

        # transpose song to C major or A minor
        transposed_song = song.transpose(interval)
        transposed_songs.append(transposed_song)

    return transposed_songs


def encode_songs_old(songs: List[m21.stream.Score], flat=True) -> List[List[str]]:
    """Encode the songs with the old LSTM format. Each midi integer value is encoded to an string. The duration is encoded as an "_".

    :param songs: list of songs
    :return: list of encoded songs
    """
    encoded_songs = []
    for song in songs:
        # Every time step is a quarter note
        time_step = 0.25

        encoded_song = []

        # Save the song in MIDI format
        # A event is a note or a rest
        # If True use all notes and rests (notes at the same time played are now played after each other)
        if flat:
            for event in song.flat.notesAndRests:
                # Notes
                if isinstance(event, m21.note.Note):
                    symbol = "n" + str(event.pitch.midi)
                # Rests
                elif isinstance(event, m21.note.Rest):
                    symbol = "r"

                # For example, if the duration of the event is 1.0 (a quarter note), we need to add 4 time steps
                # The note itself and 3 "_" symbols
                steps = int(event.duration.quarterLength / time_step)
                for step in range(steps):
                    if step == 0:
                        encoded_song.append(symbol)
                    else:
                        encoded_song.append("_")
        # If False use only the notes and rests in the first instrument
        else:
            for event in song.getElementsByClass(m21.stream.Part)[0].flat.notesAndRests:
                # Notes
                if isinstance(event, m21.note.Note):
                    symbol = "n" + str(event.pitch.midi)
                # Rests
                elif isinstance(event, m21.note.Rest):
                    symbol = "r"

                # For example, if the duration of the event is 1.0 (a quarter note), we need to add 4 time steps
                # The note itself and 3 "_" symbols
                steps = int(event.duration.quarterLength / time_step)
                for step in range(steps):
                    if step == 0:
                        encoded_song.append(symbol)
                    else:
                        encoded_song.append("_")

        # cast the encoded song to string
        encoded_songs.append(encoded_song)

    return encoded_songs


def decode_songs_old(song: str, bpm: int = 120, instrument=m21.instrument.Piano()) -> m21.stream.Stream:
    """Decode the song with the old LSTM format. Each midi integer value is encoded to an string. The duration is encoded as an "_".

    :param song: the encoded song
    :param bmp: the bpm of the song
    :return: the decoded song
    """
    song_splitted = song.split(" ")

    # Remove the "n" symbol from the notes
    song_splitted = [symbol[1:] if symbol[0] == "n" else symbol for symbol in song_splitted]

    m21_stream: m21.stream.Stream = m21.stream.Stream()
    m21_stream.append(m21.tempo.MetronomeMark(number=bpm))
    m21_stream.append(instrument)

    start_symbol = None
    step_counter = 1  # Tracks the length of one note. 1 = 1/4 note, 2 = 1/2 note, 4 = 1 whole note
    step_duration = 0.25  # The duration of one step in quarter length

    for index, symbol in enumerate(song_splitted):
        # If the symbol is a note or a rest or the end of the melody
        if symbol != "_" or index == len(song_splitted) - 1:
            # Ensure that the symbol is not the start symbol
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter  # 0.25 * 1 = 0.25, 0.25 * 2 = 0.5, 0.25 * 4 = 1

                # If the symbol is a note
                if start_symbol != "r":
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                # If the symbol is a rest
                else:
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                m21_stream.append(m21_event)
                step_counter = 1

            start_symbol = symbol

        else:
            step_counter += 1

    return m21_stream


def replace_special_tokens(
    song: List[Union[int, str]],
    replace: Union[int, str],
    tokens: List[Union[int, str]],
) -> List[Union[int, str]]:
    """Replace the special tokens in the song with the given replacement.

    :param song: the song
    :param replace: the replacement
    :param tokens: the tokens to replace
    :return: the song with replaced tokens
    """
    for symbol in song:
        if symbol in tokens:
            print(f"Replacing {symbol} with {replace}")

    return [replace if symbol in tokens else symbol for symbol in song]


def load_dataset_lstm(
    paths: List[os.PathLike],
    sequence_length: int,
    mapping: Any,
    raw_songs=False,
    corpus=True,
    flat=True,
) -> Union[List[int], List[str]]:
    """Create one big list with all songs in it. It is decoded like "n60 _ _ _" to the integer values of the mapping.

    :param paths: Path to the songs
    :param sequence_length: length of the sequences
    :param mapping: the mapping of the dataset
    :param raw_songs: if True, the songs will not merged in one big list and in string format
    :param corpus: if True, the songs will be read from the corpus. Else you need to set False
    :return: Decoded songs in a list
    """
    songs: List[m21.stream.Score] = []

    if corpus:
        for path in paths:
            songs.append(read_single_from_corpus(path))  # type: ignore
    else:
        songs = read_all(paths)  # type: ignore

    # Filter out songs with bad durations
    bad_songs = []

    for index, song in enumerate(songs):
        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in ACCEPTABLE_DURATIONS:
                bad_songs.append(index)
                break

    # Remove bad songs
    for index in sorted(bad_songs, reverse=True):
        del songs[index]

    print(f"Removed {len(bad_songs)} bad songs")
    print(f"Remaining songs: {len(songs)}")

    # transpose songs to C major
    songs = transpose_songs(songs)

    # Encode the songs
    encoded_songs = encode_songs_old(songs, flat=flat)

    if raw_songs:
        tmp_encoded_songs: List[str] = []
        for encoded_song in encoded_songs:
            tmp_encoded_songs.append(" ".join(encoded_song))
        return tmp_encoded_songs

    # Create the dataset
    song_delimiters = "/ " * sequence_length
    dataset: List[int] = []
    for song in encoded_songs:  # type: ignore
        dataset.extend(mapping.numericalize(song_delimiters))
        dataset.extend(mapping.numericalize(song))

    return dataset


def to_polyphonic_encoding(song: m21.stream.Score, mapping):
    score_arr = song_to_scorearr(song)
    encoded_arr = scorearr_to_encodedarr(score_arr)
    return encodedarr_to_indexencoding(encoded_arr, mapping)


def from_polyphonic_encoding(
    index_arr,
    mapping,
    bpm: int = 120,
    instrument=m21.instrument.Piano(),
    validate=True,
) -> m21.stream.Score:
    encoded_arr = indexarr_to_encodedarr(index_arr, mapping, validate=validate)
    score_arr = encodedarr_to_scorearr(np.array(encoded_arr))
    return scorearr_to_song(score_arr, bpm=bpm, instrument=instrument)


def song_to_scorearr(song: m21.stream.Score, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    highest_time = max(
        song.flat.getElementsByClass("Note").highestTime,
        song.flat.getElementsByClass("Chord").highestTime,
    )
    maxTimeStep = round(highest_time * sample_freq) + 1
    score_arr = np.zeros((maxTimeStep, len(song.parts), NOTE_SIZE))

    def note_data(pitch, note):
        return (
            pitch.midi,
            int(round(note.offset * sample_freq)),
            int(round(note.duration.quarterLength * sample_freq)),
        )

    for idx, part in enumerate(song.parts):
        notes = []
        for elem in part.flat:
            if isinstance(elem, m21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, m21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))

        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2]))
        for n in notes_sorted:
            if n is None:
                continue
            pitch, offset, duration = n
            if max_note_dur is not None and duration > max_note_dur:
                duration = max_note_dur
            score_arr[offset, idx, pitch] = duration
            score_arr[offset + 1 : offset + duration, idx, pitch] = VALTCONT  # Continue holding note
    return score_arr


def scorearr_to_encodedarr(chordarr, skip_last_rest=True):
    """Generate numpy array with [note,duration]

    Args:
        chordarr (_type_): one hot encoded song
        skip_last_rest (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: np.array
    """
    result = []
    wait_count = 0
    for idx, timestep in enumerate(chordarr):
        flat_time = encode_timestep(timestep)
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            if wait_count > 0:
                result.append([VALTSEP, wait_count])
            result.extend(flat_time)
            wait_count = 1
    if wait_count > 0 and not skip_last_rest:
        result.append([VALTSEP, wait_count])
    return np.array(result, dtype=int).reshape(-1, 2)  # reshaping. Just in case result is empty


def encode_timestep(timestep, note_range=PIANO_RANGE, enc_type=None):
    # inst x pitch
    notes = []
    for i, n in zip(*timestep.nonzero()):
        d = timestep[i, n]
        if d < 0:
            continue  # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]:
            continue  # must be within midi range
        notes.append([n, d, i])

    notes = sorted(notes, key=lambda x: x[0], reverse=True)  # sort by note (highest to lowest)

    if enc_type is None:
        # note, duration
        return [n[:2] for n in notes]
    if enc_type == "parts":
        # note, duration, part
        return [n for n in notes]
    if enc_type == "full":
        # note_class, duration, octave, instrument
        return [[n % 12, d, n // 12, i] for n, d, i in notes]


def encodedarr_to_indexencoding(t, mapping, seq_type=SEQType.Sentence, add_eos=False):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2:
        return [encodedarr_to_indexencoding(x, mapping, seq_type) for x in t]
    t = t.copy()

    t[:, 0] = t[:, 0] + mapping.note_range[0]
    t[:, 1] = t[:, 1] + mapping.dur_range[0]

    prefix = seq_prefix(seq_type, mapping)
    suffix = np.array([mapping.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
    return np.concatenate([prefix, t.reshape(-1), suffix])


def seq_prefix(seq_type, vocab):
    if seq_type == SEQType.Empty:
        return np.empty(0, dtype=int)
    start_token = vocab.bos_idx
    # if seq_type == SEQType.Chords:
    #     start_token = vocab.stoi[CSEQ]
    # if seq_type == SEQType.Melody:
    #     start_token = vocab.stoi[MSEQ]
    return np.array([start_token, vocab.pad_idx])


def encodedarr_to_scorearr(encoded_arr, note_size=NOTE_SIZE):
    num_instruments = 1 if len(encoded_arr.shape) <= 2 else encoded_arr.max(axis=0)[-1]

    max_len = get_len_encodedarr(encoded_arr)
    # score_arr = (steps, inst, note)
    score_arr = np.zeros((max_len, num_instruments, note_size))

    idx = 0
    for step in encoded_arr:
        n, d, i = (step.tolist() + [0])[:3]  # or n,d,i
        if n < VALTSEP:
            continue  # special token
        if n == VALTSEP:
            idx += d
            continue
        score_arr[idx, i, n] = d
    return score_arr


def get_len_encodedarr(encoded_arr):
    duration = 0
    for t in encoded_arr:
        if t[0] == VALTSEP:
            duration += t[1]
    return duration + 1


def scorearr_to_song(score_arr, sample_freq=SAMPLE_FREQ, bpm=120, instrument=m21.instrument.Piano()):
    duration = m21.duration.Duration(1.0 / sample_freq)
    stream = m21.stream.Score()
    stream.append(m21.meter.TimeSignature(TIMESIG))
    stream.append(m21.tempo.MetronomeMark(number=bpm))
    stream.append(m21.key.KeySignature(0))
    for inst in range(score_arr.shape[1]):
        p = partarr_to_song(score_arr[:, inst, :], duration, instrument=instrument)
        stream.append(p)
    stream = stream.transpose(0)
    return stream


def partarr_to_song(part_arr, duration, instrument=m21.instrument.Piano()):
    "convert instrument part to music21 chords"
    part = m21.stream.Part()
    part.append(instrument)
    part_append_duration_notes(part_arr, duration, part)  # notes already have duration calculated
    return part


def part_append_duration_notes(part_arr, duration, stream):
    "convert instrument part to music21 chords"
    for tidx, t in enumerate(part_arr):
        note_idxs = np.where(t > 0)[0]  # filter out any negative values (continuous mode)
        if len(note_idxs) == 0:
            continue
        notes = []
        for nidx in note_idxs:
            note = m21.note.Note(nidx)
            note.duration = m21.duration.Duration(part_arr[tidx, nidx] * duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            if len(g) == 1:
                stream.insert(tidx * duration.quarterLength, g[0])
            else:
                chord = m21.chord.Chord(g)
                stream.insert(tidx * duration.quarterLength, chord)
    return stream


#  combining notes with different durations into a single chord may overwrite conflicting durations. Example: aylictal/still-waters-run-deep
def group_notes_by_duration(notes):
    "separate notes into chord groups"
    keyfunc = lambda n: n.duration.quarterLength
    notes = sorted(notes, key=keyfunc)
    return [list(g) for k, g in groupby(notes, keyfunc)]


def indexarr_to_encodedarr(index_arr, mapping, validate=True):
    index_arr = fix_encodedarr_order(index_arr, mapping)

    if validate:
        index_arr = validate_indexarr(index_arr, mapping.npenc_range)

    # convert from 1d arr two 2d arr
    index_arr = index_arr.copy().reshape(-1, 2)
    if index_arr.shape[0] == 0:
        return index_arr
    index_arr[:, 0] = index_arr[:, 0] - mapping.note_range[0]
    index_arr[:, 1] = index_arr[:, 1] - mapping.dur_range[0]

    if validate:
        return validate_encodedarr(index_arr)
    return index_arr


def validate_indexarr(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    if t.shape[-1] % 2 == 1:
        t = t[..., :-1]
    return t


def validate_encodedarr(t):
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    invalid_dur_idx = (t[:, 1] < 0).argmax()

    invalid_idx = max(invalid_dur_idx, invalid_note_idx)
    if invalid_idx > 0:
        if invalid_note_idx > 0 and invalid_dur_idx > 0:
            invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print("Non midi note detected. Only returning valid portion. Index, seed", invalid_idx, t.shape)
        return t[:invalid_idx]
    return t


def fix_encodedarr_order(song: np.ndarray, mapping: Any) -> np.ndarray:
    """Fix the order of the encoded song. The order coud be note note or duration duration, but it should be note duration note duration.

    :param song: song to fix
    :param mapping: mapping of the integers to symbols
    :return: song with fixed order
    """
    # Create a copy of the song
    song_copy = song.copy()

    note_range = mapping.note_range
    dur_range = mapping.dur_range

    # validate and reorder until the song is valid
    index = 0
    while True:
        # validate note, ignore SEP
        if song_copy[index] < note_range[0] or song_copy[index] >= note_range[1]:
            if song_copy[index] != mapping.sep_idx:
                # delete the index
                song_copy = np.delete(song_copy, index)

                # restart the loop
                index = 0
                continue

        # validate duration
        if song_copy[index + 1] < dur_range[0] or song_copy[index + 1] >= dur_range[1]:
            # delete the index
            song_copy = np.delete(song_copy, index + 1)

            # restart the loop
            index = 0
            continue

        index += 2

        # check if the next index is the end of the song
        if index + 2 >= len(song_copy):
            return song_copy
