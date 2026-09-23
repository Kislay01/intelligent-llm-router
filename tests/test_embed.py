import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from embed import embed_query


def test_embed_shape():
    vec = embed_query("hello world")
    assert vec.shape == (384,)


def test_embed_consistency():
    v1 = embed_query("What is the capital of France?")
    v2 = embed_query("What is the capital of France?")
    assert (v1 == v2).all()