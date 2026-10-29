"""
ABC2Vec: Self-Supervised Learning for Folk Music Representation.

This package provides tools for learning dense vector representations
of folk music tunes from ABC notation using self-supervised learning.
"""

__version__ = "0.1.0"
__author__ = "ABC2Vec Team"

from core.tokenizer.vocabulary import ABCVocabulary
from core.tokenizer.patchifier import BarPatchifier
from core.tokenizer.transposer import ABCTransposer
from core.model.encoder import ABC2VecEncoder
from core.model.embedding import PatchEmbedding

__all__ = [
    "ABCVocabulary",
    "BarPatchifier",
    "ABCTransposer",
    "ABC2VecEncoder",
    "PatchEmbedding",
]
