import pytest
import torch

from mpb_latex_ocr.models.encoder_decoder import CNNEncoder, TransformerOCRModel


def test_deep_cnn_encoder_preserves_visual_memory_shape():
    encoder = CNNEncoder(d_model=256, channels=(32, 64, 128), depths=(2, 2, 3), dropout=0.1)
    images = torch.randn(2, 1, 128, 512)

    memory = encoder(images)

    assert memory.shape == (2, 16 * 64, 256)


def test_deep_cnn_encoder_rejects_mismatched_depths():
    with pytest.raises(ValueError, match="encoder_depths"):
        CNNEncoder(d_model=128, channels=(32, 64), depths=(2,))


def test_transformer_ocr_accepts_deep_cnn_config():
    model = TransformerOCRModel(
        vocab_size=32,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        d_model=64,
        encoder_channels=(16, 32, 64),
        encoder_depths=(2, 2, 2),
        decoder_layers=1,
        nhead=4,
        dim_feedforward=128,
        max_seq_len=32,
    )
    images = torch.randn(2, 1, 128, 512)
    decoder_input_ids = torch.ones(2, 8, dtype=torch.long)

    logits = model(images, decoder_input_ids)

    assert logits.shape == (2, 8, 32)
