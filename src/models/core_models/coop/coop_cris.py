import torch
from torch.nn import functional as F

from src.models.components.cris_model import CRIS

from .context_learner import ContextLearnerMixin


class COOPCRIS(ContextLearnerMixin, CRIS):
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """img: b, 3, h, w
        word: b, words
        word_mask: b, words
        mask: b, 1, h, w.
        """
        # padding mask used in decoder
        pad_mask = (
            ~(attention_mask.bool()) if attention_mask is not None else input_ids == 0
        )

        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(pixel_values)
        input_ids, state = self.encode_text(input_ids)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, input_ids, pad_mask)
        fq = fq.view(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        return F.interpolate(pred, self.img_size, mode="bicubic", align_corners=True)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        _self = self.backbone

        x = _self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = _self.add_context_to_input_embeddings(x)

        x = x + _self.positional_embedding[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = _self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return _self.ln_final(x)
