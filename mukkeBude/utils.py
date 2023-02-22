import os
from enum import Enum

import music21 as m21
import numpy as np

BPB = 4 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array
SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1 # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

def read_single(file_path:str) -> m21.stream.Score:
    """Convert file of a song to music21.stream.Score. Accepted file types are .mid, .krn, .abc, .mxl, .musicxml

    Args:
        file_path (str): the path to file

    Returns:
        m21.stream.Score: the converted song
    """
    return m21.converter.parse(file_path)

def read_single_from_corpus(corpus_path:str) -> m21.stream.Score:
    """Convert file in music21 corpus to music21.stream.Score. Paths to corpus files can be retrieved using e.g. paths = music21.corpus.getComposer('bach')

    Args:
        corpus_path (str): the path to the file in corpus

    Returns:
        m21.stream.Score: the converted song
    """
    return m21.corpus.parse(corpus_path)

def read_all(folder_path:str) -> list[m21.stream.Score]:
    """Converts all files in folder to list[music21.stream.Score]. Accepted file types are .mid, .krn, .abc, .mxl, .musicxml

    Args:
        folder_path (str): the path to the folder

    Returns:
        list[m21.stream.Score]: list of converted songs
    """
    songs = []
    for path, subdir, files in os.walk(folder_path):
        for file in files:
            song = read_single(os.path.join(path, file))
            songs.append(song)
    return songs

def to_polyphonic_encoding(song:m21.stream.Score, mapping):
    onehotarr = song_to_onehotarr(song)
    nparr =  onehotarr_to_nparr(onehotarr)
    return nparr_to_indexencoding(nparr, mapping)

def song_to_onehotarr(song:m21.stream.Score, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    highest_time = max(song.flat.getElementsByClass('Note').highestTime, song.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, len(song.parts), NOTE_SIZE))

    def note_data(pitch, note):
        return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))

    for idx,part in enumerate(song.parts):
        notes=[]
        for elem in part.flat:
            if isinstance(elem, m21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, m21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))

        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2]))
        for n in notes_sorted:
            if n is None: continue
            pitch,offset,duration = n
            if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
            score_arr[offset, idx, pitch] = duration
            score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
    return score_arr

def onehotarr_to_nparr(chordarr, skip_last_rest=True):
    """Generate numpy array with [note,duration]

    Args:
        chordarr (_type_): one hot encoded song
        skip_last_rest (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: np.array
    """
    result = []
    wait_count = 0
    for idx,timestep in enumerate(chordarr):
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
    return np.array(result, dtype=int).reshape(-1, 2) # reshaping. Just in case result is empty

def encode_timestep(timestep, note_range=PIANO_RANGE, enc_type=None):
    # inst x pitch
    notes = []
    for i,n in zip(*timestep.nonzero()):
        d = timestep[i,n]
        if d < 0:
            continue # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]:
            continue # must be within midi range
        notes.append([n,d,i])

    notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)

    if enc_type is None:
        # note, duration
        return [n[:2] for n in notes]
    if enc_type == 'parts':
        # note, duration, part
        return [n for n in notes]
    if enc_type == 'full':
        # note_class, duration, octave, instrument
        return [[n%12, d, n//12, i] for n,d,i in notes]

def nparr_to_indexencoding(t, mapping, seq_type=SEQType.Sentence, add_eos=False):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2:
        return [nparr_to_indexencoding(x, mapping, seq_type) for x in t]
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
