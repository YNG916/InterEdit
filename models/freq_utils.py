import torch
import math

def dct_ii(x: torch.Tensor, dim: int = 1, norm: str = "ortho") -> torch.Tensor:
    """
    DCT-II using FFT trick (torch-only, batched).
    x: (..., T, ...)
    returns same shape as x, DCT coefficients along dim.
    """
    x = x.transpose(dim, -1)  # (..., T)
    N = x.shape[-1]

    # [x0..x_{N-1}, x_{N-1}..x0] length 2N
    x_ext = torch.cat([x, x.flip(dims=[-1])], dim=-1)  # (..., 2N)

    # FFT
    X = torch.fft.rfft(x_ext, dim=-1)  # (..., N+1), complex

    # take k=0..N-1
    Xk = X[..., :N]  # (..., N)

    k = torch.arange(N, device=x.device, dtype=x.dtype)
    # exp(-j*pi*k/(2N))
    phase = torch.exp(-1j * math.pi * k / (2.0 * N))
    y = (Xk * phase).real * 2.0  # (..., N)

    if norm == "ortho":
        y[..., 0] = y[..., 0] / math.sqrt(4.0 * N) * math.sqrt(2.0)
        y[..., 1:] = y[..., 1:] / math.sqrt(2.0 * N)

    y = y.transpose(dim, -1)
    return y

def band_energy(dct_coeff: torch.Tensor, k0: int, k1: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Energy pooling over frequency band [k0, k1] inclusive along time/freq dim=1.
    dct_coeff: (B, T, D)
    return: (B, D)
    """
    band = dct_coeff[:, k0:k1+1, :]  # (B, K, D)
    # sqrt(mean(x^2)) is stable amplitude-like feature
    return torch.sqrt(torch.mean(band * band, dim=1) + eps)

def default_bands(T: int, r_low: float = 0.08, r_mid: float = 0.25, r_hi: float = 0.50):
    """
    Return (low: [0..kL], mid: [kL+1..kM], high:[kM+1..kH]).
    """
    kL = int(math.floor(r_low * T))
    kM = int(math.floor(r_mid * T))
    kH = int(math.floor(r_hi * T))
    kL = max(kL, 0)
    kM = max(kM, kL)
    kH = max(kH, kM)
    kH = min(kH, T - 1)
    return (0, kL), (kL + 1, kM), (kM + 1, kH)
