"""Model module for ABC2Vec."""

from abc2vec.model.embedding import PatchEmbedding
from abc2vec.model.encoder import (
    ABC2VecConfig,
    TransformerEncoderLayer,
    ABC2VecEncoder,
    ABC2VecModel,
)
from abc2vec.model.objectives import (
    MaskedMusicModelingLoss,
    SectionContrastiveLoss,
    TranspositionInvarianceLoss,
    VariantAwareContrastiveLoss,
    ABC2VecLoss,
)

__all__ = [
    "PatchEmbedding",
    "ABC2VecConfig",
    "TransformerEncoderLayer",
    "ABC2VecEncoder",
    "ABC2VecModel",
    "MaskedMusicModelingLoss",
    "SectionContrastiveLoss",
    "TranspositionInvarianceLoss",
    "VariantAwareContrastiveLoss",
    "ABC2VecLoss",
]
