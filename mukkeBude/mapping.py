import pickle
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

from mukkeBude.utils import *

BOS = "xxbos"
PAD = "xxpad"
EOS = "xxeos"
UNK = "[UNK]"

SEP = "xxsep"  # Used to denote end of timestep (required for polyphony). separator idx = -1 (part of notes)

WAIT_LSTM = "_"
SEP_LSTM = "/"
REST = "r"
SPACE = " "

SPECIAL_TOKS = [BOS, PAD, EOS, WAIT_LSTM, SEP_LSTM, SPACE, REST, UNK, SEP]  # Important: SEP token must be last

NOTE_TOKS = [f"n{i}" for i in range(NOTE_SIZE)]  # type: ignore
DUR_TOKS = [f"d{i}" for i in range(DUR_SIZE)]  # type: ignore
NOTE_START, NOTE_END = NOTE_TOKS[0], NOTE_TOKS[-1]
DUR_START, DUR_END = DUR_TOKS[0], DUR_TOKS[-1]


class MusicMapping:
    """A `Mapping` from tokens to ids and vice versa."""

    def __init__(self, itos: List[str]):
        """Create a `Mapping` from a list of tokens.

        :param itos: List of tokens
        """
        self.itos = itos
        self.stoi: Dict[str, int] = {v: k for k, v in enumerate(self.itos)}

    def numericalize(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to their ids.

        :param tokens: List of tokens
        :return: List of ids
        """
        return [self.stoi[w] for w in tokens]

    def textify(self, nums: List[int], sep=" ") -> List[str]:
        """Convert a list of numbers to their tokens.

        :param nums: List of ids
        :param sep: Separator between tokens, defaults to `" "` (space)
        :return: List of tokens
        """
        items = [self.itos[i] for i in nums]
        return sep.join(items) if sep is not None else items

    @property
    def pad_idx(self) -> int:
        """Get the number of the special padding token.

        :return: number of the padding token
        """
        return self.stoi[PAD]

    @property
    def bos_idx(self) -> int:
        """Get the number of the special beginning of sequence token.

        :return: number of the beginning of sequence token
        """
        return self.stoi[BOS]

    @property
    def sep_idx(self) -> int:
        """Get the number of the special separator token.

        :return: number of the separator token
        """
        return self.stoi[SEP]

    @property
    def npenc_range(self) -> Tuple[int, int]:
        """Get the range of the note and duration tokens.

        :return: range of the note and duration tokens
        """
        return (self.stoi[SEP], self.stoi[DUR_END] + 1)

    @property
    def note_range(self) -> Tuple[int, int]:
        """Get the range of the notes tokens

        :return: range of the notes tokens
        """
        return self.stoi[NOTE_START], self.stoi[NOTE_END] + 1

    @property
    def dur_range(self) -> Tuple[int, int]:
        """Get the range of the duration tokens

        :return: range of the duration tokens
        """
        return self.stoi[DUR_START], self.stoi[DUR_END] + 1

    def is_duration(self, idx: int) -> bool:
        """Check if the given index is a duration token.

        :param idx: index to check
        :return: `True` if the index is a duration token, `False` otherwise
        """
        return idx >= self.dur_range[0] and idx < self.dur_range[1]

    def is_duration_or_pad(self, idx: int) -> bool:
        """Check if the given index is a duration token or the padding token.

        :param idx: index to check
        :return: `True` if the index is a duration token or the padding token, `False` otherwise
        """
        return idx == self.pad_idx or self.is_duration(idx)

    def __getstate__(self):
        return {"itos": self.itos}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.stoi = {v: k for k, v in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save(self, path: Union[str, Path, os.PathLike]):
        """Save the `MApping` to `path`

        :param path: path to save the `Mapping` to
        """
        pickle.dump(self.itos, open(path, "wb"))
        with open(path, "w") as f:
            for line in self.itos:
                f.write(f"{line} ")

    @classmethod
    def create(cls) -> "MusicMapping":
        """Create a mapping from a fixed set of `tokens`.

        :return: `MusicMapping` object
        """
        itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS
        if len(itos) % 8 != 0:
            itos = itos + [f"dummy{i}" for i in range(len(itos) % 8)]
        return cls(itos)

    @classmethod
    def load(cls, path) -> "MusicMapping":
        """Load a `Mapping` from `path`.

        :param path: path to load the `Mapping` from
        :return: `Mapping` object
        """
        itos = pickle.load(open(path, "rb"))
        return cls(itos)
