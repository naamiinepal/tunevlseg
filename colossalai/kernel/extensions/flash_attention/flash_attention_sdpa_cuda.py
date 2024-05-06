from ..base_extension import _Extension


class FlashAttentionSdpaCudaExtension(_Extension):
    def __init__(self):
        super().__init__(
            name="flash_attention_sdpa_cuda", support_aot=False, support_jit=False
        )

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> bool:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError(
            "Flash attention SDPA does not require ahead-of-time compilation."
        )

    def build_jit(self) -> None:
        raise NotImplementedError(
            "Flash attention SDPA does not require just-in-time compilation."
        )

    def load(self):
        import torch

        def flash_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dropout_p: float = 0.0,
            scale: float | None = None,
            attention_mask: torch.Tensor | None = None,
            is_causal: bool = False,
            cu_seqlens_q: torch.Tensor | None = None,
            cu_seqlens_kv: torch.Tensor | None = None,
            max_seqlen_q: int | None = None,
            max_seqlen_kv: int | None = None,
            q_indices: torch.Tensor | None = None,
            kv_indices: torch.Tensor | None = None,
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                scale=scale,
            )

        return flash_attention
