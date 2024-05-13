from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.cris_model import CRIS

from .context_learner import CoCoOpContextLearner, CoOpContextLearner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.nn.common_types import _size_2_t

    from src.models.components.cris_model import CLIPImageOutputType


class COOPCRIS(CRIS):
    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        context_learner: type[CoOpContextLearner | CoCoOpContextLearner],
        freeze_all: bool = True,
        no_freeze_last_layer: bool = False,
        use_new_last_layer: bool = False,
        new_last_layer_kernel_size: _size_2_t = 5,
        residual_ratio: float = 0.5,
    ) -> None:
        super().__init__(**model_cfg)

        self.assign_model_learnability(
            freeze_all,
            no_freeze_last_layer,
            use_new_last_layer,
            new_last_layer_kernel_size,
            residual_ratio,
        )

        self.context_learner = context_learner(
            max_network_depth=self.backbone.transformer.layers,
            visual_dim=self.backbone.visual.output_dim,
            context_dim=self.word_dim,
            embedding_layer=self.backbone.token_embedding,
        )

        # CRIS has vision output in the form of (batch_size, output_dim, H, W)
        # So need to pool it without any learnable params to (batch_size, output_dim)
        # Since image features is not needed for CoOp, no need for an extra computation
        self.image_features_pooler_or_identity = (
            self._pool_4D_tensor
            if isinstance(self.context_learner, CoCoOpContextLearner)
            else nn.Identity()
        )

    def assign_model_learnability(
        self,
        freeze_all: bool,
        no_freeze_last_layer: bool,
        use_new_last_layer: bool,
        new_last_layer_kernel_size: _size_2_t,
        residual_ratio: float,
    ):
        if freeze_all:
            self.eval()
            self.requires_grad_(False)

        self.additive_decoder_layer = None

        if use_new_last_layer:
            intermediate_dim = 64
            self.additive_decoder_layer = nn.Sequential(
                nn.Conv2d(self.proj.in_dim * 2, intermediate_dim, 1, bias=False),
                nn.Upsample(
                    size=self.img_size,
                    mode="bilinear",
                ),
                nn.Conv2d(
                    intermediate_dim,
                    1,
                    kernel_size=new_last_layer_kernel_size,
                    padding="same",
                    padding_mode="replicate",
                ),
            )
            self.residual_ratio = nn.Parameter(torch.tensor(residual_ratio))
        elif no_freeze_last_layer:
            # Unfreeze text alignment layer
            self.proj.txt.requires_grad_(True)

            # Unfreeze visual transformation layer
            self.proj.vis[-1].requires_grad_(True)

    @staticmethod
    def _pool_4D_tensor(x: torch.Tensor) -> torch.Tensor:
        # Leave batch and channel dimension, pool from other dims
        return x.mean((2, 3))

    def get_pad_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Get pad mask from the parent method
        pad_mask = super().get_pad_mask(input_ids, attention_mask)

        # Expand the mask to include context
        return self.context_learner.update_pad_mask_for_context(
            pad_mask=pad_mask,
            max_length=self.max_length,
        )

    def text_trasnformer_forward(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = x.size(0)
        attn_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for idx, block in enumerate(self.backbone.transformer.resblocks):
            # shape: (seq_length, batch_size, d_model)
            x = block(x, *args, **kwargs, attn_mask=attn_mask)

            if idx < self.context_learner.prompt_depth:
                # shape: (batch_size, seq_length, d_model)
                x = x.movedim(0, 1)

                # Overwrite the prompts skipping the BOS Token till the prompt depth
                # shape: (batch_size, seq_length, d_model)
                self.context_learner.mutate_text_hidden_states(
                    hidden_states=x, image_features=image_features, index=idx
                )

                # shape: (seq_length, batch_size, d_model)
                x = x.movedim(1, 0)

        return x

    def encode_text(
        self,
        text: torch.Tensor,
        image_features: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _self = self.backbone

        x = _self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = self.context_learner(
            input_embeddings=x,
            max_length=self.max_length,
            image_features=image_features,
        )

        x += _self.positional_embedding[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_trasnformer_forward(x, image_features, *args, **kwargs)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = _self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        # Add indices by context number but clip to the word length
        second_indices = torch.minimum(
            text.argmax(dim=-1) + self.context_learner.num_context,
            torch.tensor(self.max_length - 1),
        )

        pooled_output = x[torch.arange(x.shape[0]), second_indices]

        state = pooled_output @ _self.text_projection

        return x, state

    def get_unimodal_outputs(
        self,
        image_input: torch.Tensor,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[CLIPImageOutputType, torch.Tensor, torch.Tensor]:
        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(image_input)

        # Get image features from the last element
        image_features = self.image_features_pooler_or_identity(vis[-1])

        input_ids, state = self.encode_text(input_ids, image_features, *args, **kwargs)
        return vis, input_ids, state

    def forward(
        self,
        text_input: Mapping[str, torch.Tensor],
        image_input: torch.Tensor,
    ):
        """img: b, 3, h, w
        word: b, words
        word_mask: b, words
        mask: b, 1, h, w.
        """
        input_ids = text_input["input_ids"]

        attention_mask = text_input.get("attention_mask")

        # padding mask used in decoder
        pad_mask = self.get_pad_mask(input_ids, attention_mask)

        vis, input_ids, state = self.get_unimodal_outputs(
            image_input,
            input_ids,
            key_padding_mask=pad_mask,
        )

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, input_ids, pad_mask)
        fq = fq.view(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        logits = F.interpolate(pred, self.img_size, mode="bicubic", align_corners=True)

        if self.additive_decoder_layer is None:
            return logits

        return (
            1 - self.residual_ratio
        ) * logits + self.residual_ratio * self.additive_decoder_layer(fq)
