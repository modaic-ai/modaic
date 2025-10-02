import pathlib

import pytest

from modaic.context import Text
from modaic.context.text import TextFile
from modaic.storage import InPlaceFileStore


def test_text_creation_and_chunking():
    t = Text(text="alpha beta gamma")
    t.chunk_text(lambda s: s.split())
    assert [c.text for c in t.chunks] == ["alpha", "beta", "gamma"]


def test_text_from_file_reads_content(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world")
    t = Text.from_file(p, type="txt")
    assert t.text == "hello world"


def test_textfile_hydration_and_dump():
    store = InPlaceFileStore("tests/artifacts/test_dir_txt")
    tf = TextFile(file_ref="file1.txt", file_type="txt")
    tf.hydrate(store)
    assert tf.dump().strip() != ""


def test_textfile_chunk_text():
    store = InPlaceFileStore("tests/artifacts/test_dir_txt")
    tf = TextFile(file_ref="file1.txt", file_type="txt")
    tf.hydrate(store)
    s = tf.dump()
    print(s)
    tf.chunk_text(lambda s: s.splitlines())
    assert len(tf.chunks) >= 1
