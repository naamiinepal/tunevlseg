import torch
from torch.nn import TransformerDecoderLayer


class PreCrossAttentionTransformerDecoderLayer(TransformerDecoderLayer):
    """Transformer Decoder Layer with cross-attention before the self-attention.

    Args:
    ----
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)

    """

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Same forward function except the change in the order of cross-attention.
        First cross-attention then self-attention.
        `norm2` applied to multi-head attention even if appleid before to have the
        consistency with the overrriden class.
        2 suffix is for multi-head attention and 1 is for self-attention.
        """
        x = tgt
        if self.norm_first:
            x += self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x += self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            return x + self._ff_block(self.norm3(x))

        x = self.norm2(
            x
            + self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            ),
        )
        x = self.norm1(
            x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal),
        )
        return self.norm3(x + self._ff_block(x))
