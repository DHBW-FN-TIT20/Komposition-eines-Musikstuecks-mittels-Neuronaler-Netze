import music21 as m21
import os

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