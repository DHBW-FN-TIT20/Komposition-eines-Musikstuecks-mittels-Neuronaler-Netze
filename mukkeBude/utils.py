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

def write_midi(song:m21.stream.Score, output_path:str="test.mid") -> None:
    song.write('midi', fp=output_path)
    
def write_musicxml(song:m21.stream.Score, output_path:str="test.mid") -> None:
    song.write('musicxml', fp=output_path)

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
    score_arr = song_to_scorearr(song)
    encoded_arr =  scorearr_to_encodedarr(score_arr)
    return encodedarr_to_indexencoding(encoded_arr, mapping)

def from_polyphonic_encoding(index_arr, mapping, bpm:int=120) -> m21.stream.Score:
    encoded_arr = indexarr_to_encodedarr(index_arr, mapping)
    score_arr = encodedarr_to_scorearr(np.array(encoded_arr))
    return scorearr_to_song(score_arr, bpm=bpm)

def song_to_scorearr(song:m21.stream.Score, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
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
        n,d,i = (step.tolist()+[0])[:3] # or n,d,i
        if n < VALTSEP: continue # special token
        if n == VALTSEP:
            idx += d
            continue
        score_arr[idx,i,n] = d
    return score_arr

def get_len_encodedarr(encoded_arr):
    duration = 0
    for t in encoded_arr:
        if t[0] == VALTSEP: 
            duration += t[1]
    return duration + 1

def scorearr_to_song(score_arr, sample_freq=SAMPLE_FREQ, bpm=120):
    duration = m21.duration.Duration(1. / sample_freq)
    stream = m21.stream.Score()
    stream.append(m21.meter.TimeSignature(TIMESIG))
    stream.append(m21.tempo.MetronomeMark(number=bpm))
    stream.append(m21.key.KeySignature(0))
    for inst in range(score_arr.shape[1]):
        p = partarr_to_song(score_arr[:,inst,:], duration)
        stream.append(p)
    stream = stream.transpose(0)
    return stream

def partarr_to_song(part_arr, duration):
    "convert instrument part to music21 chords"
    part = m21.stream.Part()
    part.append(m21.instrument.Piano()) #TODO hier kann man das Instrument mitgeben
    part_append_duration_notes(part_arr, duration, part) # notes already have duration calculated
    return part

def part_append_duration_notes(part_arr, duration, stream):
    "convert instrument part to music21 chords"
    for tidx,t in enumerate(part_arr):
        note_idxs = np.where(t > 0)[0] # filter out any negative values (continuous mode)
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = m21.note.Note(nidx)
            note.duration = m21.duration.Duration(part_arr[tidx,nidx]*duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            if len(g) == 1:
                stream.insert(tidx*duration.quarterLength, g[0])
            else:
                chord = m21.chord.Chord(g)
                stream.insert(tidx*duration.quarterLength, chord)
    return stream

from itertools import groupby
#  combining notes with different durations into a single chord may overwrite conflicting durations. Example: aylictal/still-waters-run-deep
def group_notes_by_duration(notes):
    "separate notes into chord groups"
    keyfunc = lambda n: n.duration.quarterLength
    notes = sorted(notes, key=keyfunc)
    return [list(g) for k,g in groupby(notes, keyfunc)]

def indexarr_to_encodedarr(index_arr, mapping, validate=True):
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
        if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
        return t[:invalid_idx]
    return t