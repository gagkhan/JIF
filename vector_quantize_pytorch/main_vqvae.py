import torch
from vector_quantize_pytorch import VectorQuantize

if __name__ == "__main__":

    vq = VectorQuantize(
        dim=256,
        codebook_size=512,
        decay=0.8,
        commitment_weight=1.0,
    )

    x = torch.randn(1, 1024, 256)

    quantized, indices, commit_loss = vq(x)  # (1, 1024, 256), (1, 1024), (1)
