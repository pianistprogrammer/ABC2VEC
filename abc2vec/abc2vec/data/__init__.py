"""Data module for ABC2Vec."""

from abc2vec.data.pipeline import (
    download_and_process_dataset,
    normalize_abc,
    deduplicate_tunes,
    extract_sections,
    parse_key_signature,
    infer_tune_type,
)
from abc2vec.data.dataset import (
    ABC2VecDataset,
    SectionPairDataset,
    VariantDataset,
    load_processed_data,
)

__all__ = [
    "download_and_process_dataset",
    "normalize_abc",
    "deduplicate_tunes",
    "extract_sections",
    "parse_key_signature",
    "infer_tune_type",
    "ABC2VecDataset",
    "SectionPairDataset",
    "VariantDataset",
    "load_processed_data",
]
