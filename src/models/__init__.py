import torch

# Use tensor cores on Tensor-enabled GPUs
# Increases throughput with reduced precision
torch.set_float32_matmul_precision("medium")
