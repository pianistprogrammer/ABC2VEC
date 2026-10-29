---
license: cc-by-4.0
task_categories:
- other
language:
- en
tags:
- music
- folk-music
- irish-traditional-music
- abc-notation
- symbolic-music
size_categories:
- 100K<n<1M
---

# ABC2Vec Irish Folk Music Dataset

This dataset contains 211,524 Irish traditional tunes in ABC notation, preprocessed and split for training representation learning models.

## Dataset Description

- **Curated by:** IrishMAN Dataset (The Session + ABCnotation.com)
- **Processed for:** ABC2Vec: Self-Supervised Representation Learning for Irish Folk Music
- **Language:** ABC notation (symbolic music format)
- **License:** CC-BY-4.0

## Dataset Structure

### Data Splits

| Split | Tunes | File Size |
|-------|-------|-----------|
| Train | 198,893 | 70 MB |
| Validation | 10,469 | 3.7 MB |
| Test | 2,162 | 778 KB |
| **Total** | **211,524** | **~74 MB** |

### Data Fields

Each tune contains:
- `tune_id`: Unique identifier
- `title`: Tune name
- `abc_body`: ABC notation of the melody
- `tune_type`: Rhythmic category (jig, reel, polka, waltz, etc.)
- `mode`: Tonal mode (major, minor, dorian, mixolydian)
- `key`: Key signature
- `meter`: Time signature
- `bar_count`: Number of bars in the tune

### Dataset Statistics

- **Tune Types:** 44.9% reels, 21.3% jigs, 14.5% polkas, 12.2% waltzes
- **Modes:** 80.2% major, 11.3% minor, 5.4% Dorian, 3.0% Mixolydian
- **Keys:** 30.5% G, 26.8% D, 13.9% A (sharp keys dominant)
- **Median Length:** 18 bars, 287 characters

## Usage

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("pianistprogrammer/abc2vec-irish-folk-dataset")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# Example tune
print(train[0]["abc_body"])
print(f"Type: {train[0]['tune_type']}, Mode: {train[0]['mode']}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{abc2vec2025,
  title={ABC2Vec: Self-Supervised Representation Learning for Irish Folk Music},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## Source

This dataset is derived from:
- **The Session** (thesession.org): Community-maintained Irish traditional music archive
- **ABCnotation.com**: Long-standing ABC notation repository

Processed as part of the IrishMAN (Irish Music ABC Notation) corpus.

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0)

The original tunes are traditional folk music in the public domain. This processed dataset is released under CC-BY-4.0.

## Additional Files

- `vocab.json`: Character vocabulary for tokenization (98 tokens)
- `metadata.csv`: Complete metadata for all 211,524 tunes

## Contact

For questions or issues with this dataset, please open an issue on the [ABC2Vec GitHub repository](https://github.com/pianistprogrammer/ABC2VEC).
