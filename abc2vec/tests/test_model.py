"""Unit tests for ABC2Vec model components."""

import pytest
import torch

from abc2vec.model import (
    ABC2VecConfig,
    ABC2VecModel,
    PatchEmbedding,
    TransformerEncoderLayer,
)
from abc2vec.tokenizer import ABCVocabulary


class TestPatchEmbedding:
    """Test cases for PatchEmbedding layer."""

    @pytest.fixture
    def patch_embed(self):
        """Create PatchEmbedding fixture."""
        return PatchEmbedding(
            vocab_size=100,
            d_char=64,
            d_model=256,
            max_bar_length=64,
            max_bars=64,
            pad_idx=0,
        )

    def test_forward_shape(self, patch_embed):
        """Test output shapes."""
        batch_size = 4
        bar_indices = torch.randint(0, 100, (batch_size, 64, 64))
        char_mask = torch.ones(batch_size, 64, 64, dtype=torch.bool)
        bar_mask = torch.ones(batch_size, 64, dtype=torch.bool)

        embeddings, mask = patch_embed(bar_indices, char_mask, bar_mask)

        assert embeddings.shape == (batch_size, 64, 256)
        assert mask.shape == (batch_size, 64)

    def test_masking(self, patch_embed):
        """Test that padding is properly masked."""
        batch_size = 2
        bar_indices = torch.randint(0, 100, (batch_size, 64, 64))
        char_mask = torch.ones(batch_size, 64, 64, dtype=torch.bool)
        bar_mask = torch.ones(batch_size, 64, dtype=torch.bool)

        # Mask out second half of bars
        bar_mask[:, 32:] = False

        embeddings, mask = patch_embed(bar_indices, char_mask, bar_mask)

        # Mask should match input
        assert torch.equal(mask, bar_mask)


class TestTransformerEncoderLayer:
    """Test cases for TransformerEncoderLayer."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        layer = TransformerEncoderLayer(d_model=256, n_heads=8, d_ff=1024)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, 256)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = layer(x, mask)

        assert output.shape == x.shape

    def test_without_mask(self):
        """Test forward pass without mask."""
        layer = TransformerEncoderLayer(d_model=256, n_heads=8, d_ff=1024)

        x = torch.randn(4, 32, 256)
        output = layer(x, mask=None)

        assert output.shape == x.shape


class TestABC2VecModel:
    """Test cases for complete ABC2VecModel."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and dummy data."""
        config = ABC2VecConfig(
            vocab_size=100,
            max_bar_length=64,
            max_bars=64,
            d_model=256,
            n_layers=2,
        )
        model = ABC2VecModel(config)

        batch_size = 4
        bar_indices = torch.randint(0, 100, (batch_size, 64, 64))
        char_mask = torch.ones(batch_size, 64, 64, dtype=torch.bool)
        bar_mask = torch.ones(batch_size, 64, dtype=torch.bool)

        return model, bar_indices, char_mask, bar_mask

    def test_forward(self, model_and_data):
        """Test full forward pass."""
        model, bar_indices, char_mask, bar_mask = model_and_data

        output = model(bar_indices, char_mask, bar_mask)

        assert "bar_embeddings" in output
        assert "tune_embedding" in output
        assert output["tune_embedding"].shape == (4, 128)

    def test_mmm_forward(self, model_and_data):
        """Test Masked Music Modeling forward pass."""
        model, bar_indices, char_mask, bar_mask = model_and_data

        mmm_out = model.forward_mmm(bar_indices, char_mask, bar_mask)

        assert "mmm_logits" in mmm_out
        assert "mmm_targets" in mmm_out
        assert "mmm_mask" in mmm_out

    def test_contrastive_forward(self, model_and_data):
        """Test contrastive forward pass."""
        model, bar_indices, char_mask, bar_mask = model_and_data

        emb1, emb2 = model.forward_contrastive(
            bar_indices, char_mask, bar_mask,
            bar_indices, char_mask, bar_mask,
        )

        assert emb1.shape == (4, 128)
        assert emb2.shape == (4, 128)

        # Embeddings should be L2-normalized
        norms = torch.norm(emb1, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_get_embedding(self, model_and_data):
        """Test inference-time embedding extraction."""
        model, bar_indices, char_mask, bar_mask = model_and_data

        model.eval()
        with torch.no_grad():
            embedding = model.get_embedding(bar_indices, char_mask, bar_mask)

        assert embedding.shape == (4, 128)

    def test_save_load(self, model_and_data, tmp_path):
        """Test saving and loading model."""
        model, bar_indices, char_mask, bar_mask = model_and_data

        # Save
        checkpoint_path = tmp_path / "model.pt"
        model.save_pretrained(str(checkpoint_path))

        # Load
        loaded_model = ABC2VecModel.load_pretrained(str(checkpoint_path))

        # Compare outputs
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            emb1 = model.get_embedding(bar_indices, char_mask, bar_mask)
            emb2 = loaded_model.get_embedding(bar_indices, char_mask, bar_mask)

        assert torch.allclose(emb1, emb2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
