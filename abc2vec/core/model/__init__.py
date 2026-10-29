"""Model module for ABC2Vec."""

from core.model.embedding import PatchEmbedding
from core.model.encoder import (
    ABC2VecConfig,
    TransformerEncoderLayer,
    ABC2VecEncoder,
    ABC2VecModel,
)
from core.model.objectives import (
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
