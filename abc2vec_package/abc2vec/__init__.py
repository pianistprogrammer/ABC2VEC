"""
ABC2Vec: Self-Supervised Learning for Folk Music Representation.

This package provides tools for learning dense vector representations
of folk music tunes from ABC notation using self-supervised learning.
"""

__version__ = "0.1.0"
__author__ = "ABC2Vec Team"

from abc2vec.tokenizer.vocabulary import ABCVocabulary
from abc2vec.tokenizer.patchifier import BarPatchifier
from abc2vec.tokenizer.transposer import ABCTransposer
from abc2vec.model.encoder import ABC2VecEncoder
from abc2vec.model.embedding import PatchEmbedding

__all__ = [
    "ABCVocabulary",
    "BarPatchifier",
    "ABCTransposer",
    "ABC2VecEncoder",
    "PatchEmbedding",
]
