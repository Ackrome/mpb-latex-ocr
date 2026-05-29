"""Compact CNN encoder plus Transformer decoder baseline."""

from __future__ import annotations

import torch
from torch import nn


class CNNResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.network = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activation(images + self.network(images))


class CNNEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        channels: list[int] | tuple[int, ...] = (32, 64, 128),
        depths: list[int] | tuple[int, ...] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        depths = tuple(depths or (1,) * len(channels))
        if len(depths) != len(channels):
            raise ValueError(
                f"encoder_depths must match encoder_channels length: {len(depths)} != "
                f"{len(channels)}"
            )

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, depth in zip(channels, depths, strict=True):
            if depth < 1:
                raise ValueError("CNN encoder stage depths must be >= 1.")
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            for _ in range(depth - 1):
                layers.append(CNNResidualBlock(out_channels, dropout=dropout * 0.5))
            in_channels = out_channels

        layers.extend(
            [
                nn.Conv2d(in_channels, d_model, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        features = features.flatten(2).transpose(1, 2)
        return self.proj(features)


class TransformerOCRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        d_model: int = 256,
        encoder_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        encoder_depths: list[int] | tuple[int, ...] | None = None,
        decoder_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_seq_len = max_seq_len

        self.encoder = CNNEncoder(
            d_model=d_model,
            channels=encoder_channels,
            depths=encoder_depths,
            dropout=dropout,
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        if decoder_input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Decoder sequence length {decoder_input_ids.size(1)} exceeds max_seq_len "
                f"{self.max_seq_len}."
            )

        memory = self.encoder(images)
        positions = torch.arange(
            decoder_input_ids.size(1),
            device=decoder_input_ids.device,
            dtype=torch.long,
        )
        positions = positions.unsqueeze(0).expand_as(decoder_input_ids)
        target = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(
                decoder_input_ids.size(1),
                decoder_input_ids.size(1),
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = decoder_input_ids.eq(self.pad_id)
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.output(self.norm(decoded))

    @torch.inference_mode()
    def generate(self, images: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        max_length = int(max_length or self.max_seq_len)
        batch_size = images.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=self.bos_id,
            dtype=torch.long,
            device=images.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        for _ in range(max_length - 1):
            logits = self.forward(images, generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(self.eos_id)
            if bool(finished.all()):
                break

        return generated
