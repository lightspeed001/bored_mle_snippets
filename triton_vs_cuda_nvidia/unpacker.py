def reference_dequant(packed: torch.Tensor,
                      scales: torch.Tensor,
                      n: int,
                      k: int,
                      group: int = 128) -> torch.Tensor:
    """Return an FP32 dequantized [N, K] matrix for correctness testing."""
    u = packed.to(torch.int16)                  # Avoid signed-shift surprises.
    lo = u & 0x0F
    hi = (u >> 4) & 0x0F

    # Convert unsigned nibbles to signed two's-complement INT4.
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)

    # Adjacent low/high nibbles become adjacent logical K elements.
    q = torch.stack((lo, hi), dim=-1).reshape(n, k)
    q = q.reshape(n, k // group, group).to(torch.float32)
    return (q * scales.to(torch.float32).unsqueeze(-1)).reshape(n, k)
