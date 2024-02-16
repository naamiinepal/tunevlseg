import torch

# By default, TF32 tensor cores are disabled for matrix multiplications and enabled for
# convolutions, although most neural network workloads have the same convergence
# behavior when using TF32 as they have with fp32.
torch.set_float32_matmul_precision("medium")

# Use tensor cores on Tensor-enabled GPUs
# Increases throughput with reduced precision
